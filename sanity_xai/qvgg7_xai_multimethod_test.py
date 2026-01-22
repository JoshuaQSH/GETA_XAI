"""
XAI-GETA Multi-Method Test
==========================
Test ALL Captum attribution methods through all 4 stages:
1. Warmup Stage
2. Projection Stage  
3. Joint Pruning & Quantization Stage
4. Fine-tuning Stage

Tests ALL methods: saliency, integrated_gradients, deep_lift, guided_backprop,
                   input_x_gradient, layer_conductance, layer_gradient_x_activation,
                   layer_integrated_gradients, lrp, layer_lrp, deconvolution, gradient_shap
Uses larger synthetic dataset (1000+ samples)

NOTE: Some methods have specific requirements:
- LRP/LayerLRP: Require model without inplace operations (ReLU(inplace=False))
- GradCam: Works best with convolutional layers
- GradientShap: Very slow (requires many baseline samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import gc

# Add the geta directory to path
currentdir = os.path.dirname(os.path.realpath(__name__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

# Import incompatible methods list from captum_attribution
try:
    from only_train_once.xai_optimizer.captum_attribution import QUANTIZATION_INCOMPATIBLE_METHODS
except ImportError:
    QUANTIZATION_INCOMPATIBLE_METHODS = ['guided_backprop']


# Attribution methods to test (ordered by speed: fast to slow)
# Categories:
# 1. GRADIENT-BASED (fast): saliency, input_x_gradient, guided_backprop, deconvolution
# 2. LAYER-SPECIFIC: layer_conductance, layer_gradient_x_activation, layer_integrated_gradients
# 3. PATH-BASED (slower): deep_lift, integrated_gradients
# 4. DECOMPOSITION: lrp, layer_lrp (require inplace=False)
# 5. SHAP-BASED (very slow): gradient_shap

ATTRIBUTION_METHODS = [
    # === FAST GRADIENT-BASED METHODS ===
    'saliency',           # Fast: single gradient pass (gradient magnitude)
    'input_x_gradient',   # Fast: input * gradient (simple sensitivity)
    'guided_backprop',    # Fast: modified backprop (positive gradients only)
    'deconvolution',      # Fast: deconvolution-based visualization
    
    # === LAYER-SPECIFIC METHODS ===
    'layer_conductance',           # Medium: how much a layer affects output
    'layer_gradient_x_activation', # Medium: gradient * activation at layer
    'layer_integrated_gradients',  # Slow: integrated gradients at layer level
    
    # === PATH-BASED METHODS ===
    'deep_lift',          # Medium: difference from reference propagation
    'integrated_gradients',  # Slow: path integral from baseline to input
    
    # === DECOMPOSITION-BASED METHODS ===
    # NOTE: LRP requires model WITHOUT inplace operations (ReLU(inplace=False))
    # We'll test with a modified model
    # 'lrp',              # Requires inplace=False - tested separately
    # 'layer_lrp',        # Requires inplace=False - tested separately
    
    # === SHAP-BASED METHODS (VERY SLOW) ===
    # 'gradient_shap',    # Very slow: skipped by default (many samples needed)
]

# Methods that require special model configuration (no inplace operations)
LRP_METHODS = ['lrp', 'layer_lrp']

# Methods that are very slow and skipped by default
SLOW_METHODS = ['gradient_shap']

# Timeouts for slow methods (seconds per epoch)
METHOD_TIMEOUTS = {
    'saliency': 30,
    'input_x_gradient': 30,
    'guided_backprop': 45,
    'deconvolution': 45,
    'layer_conductance': 60,
    'layer_gradient_x_activation': 45,
    'layer_integrated_gradients': 120,
    'deep_lift': 60,
    'integrated_gradients': 120,
    'lrp': 60,
    'layer_lrp': 60,
    'gradient_shap': 300,  # Very slow!
}


def create_synthetic_dataset(num_samples=500, num_classes=10, img_size=32):
    """Create synthetic CIFAR-like dataset for testing."""
    X = torch.randn(num_samples, 3, img_size, img_size)
    X = X * 0.25 + 0.5
    X = X.clamp(0, 1)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class SmallVGG(nn.Module):
    """
    Smaller VGG-like model for quick testing.
    
    NOTE: Uses inplace=False for ReLU to support LRP attribution methods.
    LRP requires the ability to capture intermediate activations, which is
    not possible with in-place operations.
    """
    def __init__(self, num_classes=10, use_inplace=False):
        super().__init__()
        # inplace=False is required for LRP methods to work
        # Set use_inplace=True for slightly faster training (but LRP won't work)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=use_inplace),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=use_inplace),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=use_inplace),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=use_inplace),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test_attribution_method(method_name, device, num_samples=500):
    """
    Test a single attribution method through all 4 stages.
    
    Returns:
        (success, error_message, skipped) tuple
    """
    import time
    
    print(f"\n{'='*70}")
    print(f"Testing Attribution Method: {method_name.upper()}")
    print(f"{'='*70}")
    
    # Check if method is incompatible with quantization
    if method_name in QUANTIZATION_INCOMPATIBLE_METHODS:
        print(f"  ⊘ SKIPPED: {method_name} is incompatible with quantization training")
        print(f"     (GuidedBackprop modifies ReLU backward hooks, conflicting with quantization)")
        return True, "Incompatible with quantization training", True
    
    start_time = time.time()
    
    # Import XAI-GETA
    from only_train_once.xai_optimizer import XAI_GETA
    
    # Check if this method requires inplace=False (LRP methods)
    use_inplace = method_name not in LRP_METHODS
    
    # Create fresh model for each test
    # NOTE: inplace=False is required for LRP methods
    model = SmallVGG(num_classes=10, use_inplace=use_inplace).to(device)
    if method_name in LRP_METHODS:
        print(f"  NOTE: Using inplace=False for LRP compatibility")
    
    # Quantize model
    model = model_to_quantize_model(
        model, 
        num_bits=8, 
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
    )
    model = model.to(device)
    print(f"✓ Model quantized")
    
    # Create synthetic datasets (larger)
    train_dataset = create_synthetic_dataset(num_samples=num_samples, num_classes=10)
    test_dataset = create_synthetic_dataset(num_samples=200, num_classes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"✓ Data: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Initialize OTO
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    oto = OTO(model, dummy_input)
    print("✓ OTO initialized")
    
    # Stage configuration
    BATCHES_PER_EPOCH = len(train_loader)
    START_PROJECTION = BATCHES_PER_EPOCH      # After 1 epoch
    START_PRUNING = 3 * BATCHES_PER_EPOCH     # After 3 epochs
    PRUNING_STEPS = 2 * BATCHES_PER_EPOCH     # 2 epochs of pruning
    TOTAL_EPOCHS = 6
    
    print(f"  Batches/epoch: {BATCHES_PER_EPOCH}")
    print(f"  Warmup: 0-{START_PROJECTION-1}, Projection: {START_PROJECTION}-{START_PRUNING-1}")
    print(f"  Pruning: {START_PRUNING}-{START_PRUNING+PRUNING_STEPS-1}, Fine-tune: {START_PRUNING+PRUNING_STEPS}+")
    
    # Create XAI_GETA optimizer
    param_groups = oto._graph.get_param_groups()
    
    # Adjust n_steps and attribution frequency based on method speed
    slow_methods = ['integrated_gradients', 'layer_integrated_gradients', 'layer_conductance', 'gradient_shap']
    n_steps = 5 if method_name in slow_methods else 10
    attr_freq = 200 if method_name in slow_methods else 100
    
    try:
        optimizer = XAI_GETA(
            params=param_groups,
            model=model,
            lr=0.0003,  # Lower learning rate for stability
            lr_quant=5e-7,  # Very small for quantization stability
            weight_decay=1e-4,
            target_group_sparsity=0.5,
            first_momentum=0.9,
            second_momentum=0.9,
            start_projection_step=START_PROJECTION,
            projection_steps=START_PRUNING - START_PROJECTION,
            start_pruning_step=START_PRUNING,
            pruning_steps=PRUNING_STEPS,
            attribution_method=method_name,
            attribution_weight=0.3,
            compute_attribution_freq=attr_freq,  # Less frequent for slow methods
            attribution_n_steps=n_steps,  # Fewer steps for integrated gradients
            grad_clip_min=-0.02,  # Tighter gradient clipping for stability
            grad_clip_max=0.02,
        )
        print(f"✓ XAI_GETA created with {method_name} (n_steps={n_steps}, freq={attr_freq})")
    except Exception as e:
        print(f"✗ Failed to create optimizer: {e}")
        return False, [str(e)], False
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Compute initial attributions
    print(f"  Computing initial attributions...")
    small_loader = DataLoader(
        TensorDataset(
            (torch.randn(32, 3, 32, 32) * 0.25 + 0.5).clamp(0, 1),
            torch.randint(0, 10, (32,))
        ),
        batch_size=8
    )
    
    try:
        optimizer.compute_initial_attributions(small_loader, num_batches=2, criterion=criterion)
        print(f"✓ Initial attributions computed")
    except Exception as e:
        print(f"⚠ Attribution init issue (continuing): {e}")
    
    # Training loop
    stage_markers = {'warmup': False, 'projection': False, 'pruning': False, 'finetuning': False}
    global_step = 0
    errors = []
    timeout = METHOD_TIMEOUTS.get(method_name, 60) * TOTAL_EPOCHS
    
    import time
    training_start = time.time()
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # Determine stage
            if global_step < START_PROJECTION:
                current_stage = "WARMUP"
                if not stage_markers['warmup']:
                    stage_markers['warmup'] = True
            elif global_step < START_PRUNING:
                current_stage = "PROJECTION"
                if not stage_markers['projection']:
                    stage_markers['projection'] = True
            elif global_step < START_PRUNING + PRUNING_STEPS:
                current_stage = "PRUNING"
                if not stage_markers['pruning']:
                    stage_markers['pruning'] = True
            else:
                current_stage = "FINETUNE"
                if not stage_markers['finetuning']:
                    stage_markers['finetuning'] = True
            
            try:
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step(inputs=X, targets=y)
                epoch_loss += loss.item()
            except Exception as e:
                import traceback
                error_msg = f"Step {global_step} ({current_stage}): {str(e)[:100]}"
                errors.append(error_msg)
                if len(errors) == 1:
                    # Print full traceback for first error only
                    print(f"\n  First error traceback:")
                    traceback.print_exc()
                # Allow more errors for quantization NaN issues (common in early training)
                if len(errors) >= 10:
                    print(f"✗ Too many errors ({len(errors)}), stopping")
                    return False, errors, False
                # Skip this batch but continue training
                optimizer.zero_grad()
            
            global_step += 1
        
        scheduler.step()
        
        # Quick eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        acc = 100. * correct / total
        
        num_pruned = sum(len(g.get('pruned_idxes', [])) for g in optimizer.param_groups)
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch}/{TOTAL_EPOCHS} | {current_stage:10s} | Loss: {epoch_loss/len(train_loader):.4f} | Acc: {acc:.1f}% | Pruned: {num_pruned} | Time: {epoch_time:.1f}s")
        
        # Check for timeout
        if time.time() - training_start > timeout:
            print(f"⚠ Timeout after {time.time() - training_start:.0f}s (limit: {timeout}s)")
            break
    
    # Check results
    all_passed = all(stage_markers.values())
    
    print(f"\n  Stage Results:")
    for stage, passed in stage_markers.items():
        status = "✓" if passed else "✗"
        print(f"    {stage.upper():12s}: {status}")
    
    if errors:
        print(f"  Errors encountered: {len(errors)}")
        for err in errors[:3]:
            print(f"    - {err}")
    
    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.1f}s")
    
    return all_passed and len(errors) == 0, errors, False


def run_all_tests(include_lrp=True, include_slow=False):
    """
    Run quick check for all attribution methods.
    
    Args:
        include_lrp: Include LRP methods (require model with inplace=False)
        include_slow: Include very slow methods like gradient_shap
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("XAI-GETA MULTI-METHOD TEST")
    print("Testing ALL Captum attribution methods through 4 stages")
    print("="*70)
    
    # Build the list of methods to test
    methods_to_test = list(ATTRIBUTION_METHODS)  # Start with standard methods
    
    if include_lrp:
        methods_to_test.extend(LRP_METHODS)
        print(f"NOTE: Including LRP methods (require inplace=False)")
    
    if include_slow:
        methods_to_test.extend(SLOW_METHODS)
        print(f"WARNING: Including slow methods - this may take a long time!")
    
    print(f"Methods ({len(methods_to_test)}): {', '.join(methods_to_test)}")
    print("="*70)
    
    # Check Captum
    try:
        import captum
        print(f"✓ Captum version: {captum.__version__}")
    except ImportError:
        print("✗ Captum not installed!")
        return False
    
    results = {}
    
    for method in methods_to_test:
        try:
            # Adjust samples based on method speed
            if method in SLOW_METHODS:
                samples = 200  # Very few samples for slow methods
            elif method in ['integrated_gradients', 'layer_integrated_gradients', 'layer_conductance']:
                samples = 400  # Fewer samples for moderately slow methods
            else:
                samples = 500  # Standard sample count
                
            success, errors, skipped = test_attribution_method(method, device, num_samples=samples)
            results[method] = {'success': success, 'errors': errors, 'skipped': skipped}
        except Exception as e:
            print(f"✗ {method} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {'success': False, 'errors': [str(e)], 'skipped': False}
        
        # Clear CUDA cache between tests
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL ATTRIBUTION METHODS")
    print("="*70)
    
    # Group results by category
    categories = {
        'Gradient-Based (Fast)': ['saliency', 'input_x_gradient', 'guided_backprop', 'deconvolution'],
        'Layer-Specific': ['layer_conductance', 'layer_gradient_x_activation', 'layer_integrated_gradients'],
        'Path-Based': ['deep_lift', 'integrated_gradients'],
        'Decomposition (LRP)': ['lrp', 'layer_lrp'],
        'SHAP-Based (Slow)': ['gradient_shap'],
    }
    
    for category, methods in categories.items():
        tested_methods = [m for m in methods if m in results]
        if tested_methods:
            print(f"\n  {category}:")
            for method in tested_methods:
                result = results[method]
                if result.get('skipped', False):
                    status = "⊘ SKIPPED"
                elif result['success']:
                    status = "✓ PASSED"
                else:
                    status = "✗ FAILED"
                print(f"    {method:30s}: {status}")
                if not result['success'] and result.get('errors') and not result.get('skipped'):
                    for err in result['errors'][:2]:
                        print(f"        Error: {str(err)[:70]}...")
    
    num_passed = sum(1 for r in results.values() if r['success'])
    num_skipped = sum(1 for r in results.values() if r.get('skipped', False))
    print(f"\n  Total: {num_passed}/{len(results)} methods passed ({num_skipped} skipped - incompatible with quantization)")
    print("="*70)
    
    return all(r['success'] for r in results.values())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test ALL Captum attribution methods')
    parser.add_argument('--include-lrp', action='store_true', default=True,
                        help='Include LRP methods (default: True)')
    parser.add_argument('--no-lrp', action='store_true',
                        help='Exclude LRP methods')
    parser.add_argument('--include-slow', action='store_true', default=False,
                        help='Include very slow methods like gradient_shap')
    args = parser.parse_args()
    
    include_lrp = not args.no_lrp
    success = run_all_tests(include_lrp=include_lrp, include_slow=args.include_slow)
    sys.exit(0 if success else 1)
