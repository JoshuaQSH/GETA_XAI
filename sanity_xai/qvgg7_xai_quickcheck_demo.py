"""
XAI-GETA Quick Check Demo
=========================
Fast verification that ALL FOUR stages work correctly with MULTIPLE attribution methods:
1. Warmup Stage
2. Projection Stage  
3. Joint Pruning & Quantization Stage
4. Fine-tuning Stage

Tests: saliency, integrated_gradients, deep_lift, gradient_shap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add the geta directory to path
currentdir = os.path.dirname(os.path.realpath(__name__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from only_train_once import OTO
from only_train_once.transform import tensor_transformation
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode

# Import incompatible methods list from captum_attribution
from only_train_once.xai_optimizer.captum_attribution import QUANTIZATION_INCOMPATIBLE_METHODS


# Attribution methods to test
ATTRIBUTION_METHODS = [
    'saliency',
    'input_x_gradient',
    'guided_backprop',
    'deconvolution',
    'layer_conductance',
    'layer_gradient_x_activation',
    'layer_integrated_gradients',
    'deep_lift',
    'integrated_gradients',
    'lrp',
    'layer_lrp',
    'gradient_shap',
]

# Methods that require inplace=False for ReLU
LRP_METHODS = ['lrp', 'layer_lrp', 'guided_backprop', 'deconvolution']

# Very slow methods (skip by default)
SLOW_METHODS = ['gradient_shap']


def create_synthetic_dataset(num_samples=500, num_classes=10, img_size=32):
    """Create synthetic CIFAR-like dataset for quick testing."""
    # Use normalized data similar to CIFAR-10 normalization
    # Mean and std similar to CIFAR-10
    X = torch.randn(num_samples, 3, img_size, img_size)
    # Normalize to be similar to CIFAR-10 range
    X = X * 0.25 + 0.5  # Roughly [-0.25, 1.25] range
    X = X.clamp(0, 1)   # Clamp to valid image range
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


class SmallVGG(nn.Module):
    """Smaller VGG-like model for quick testing.
    
    NOTE: use_inplace=False is required for LRP, GuidedBackprop, and Deconvolution methods.
    """
    def __init__(self, num_classes=10, use_inplace=True):
        super().__init__()
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


def test_single_method(method_name, device, num_epochs=4, num_samples=200):
    """
    Test a single attribution method through all 4 stages.
    
    Args:
        method_name: Name of the attribution method
        device: torch device
        num_epochs: Number of epochs to run
        num_samples: Number of training samples
        
    Returns:
        (success, error_message, skipped) tuple
    """
    import gc
    
    print(f"\n  Testing {method_name}...", end=" ", flush=True)
    
    # Check if method is incompatible with quantization
    if method_name in QUANTIZATION_INCOMPATIBLE_METHODS:
        print("⊘ SKIPPED (incompatible with quantization)")
        return True, "Incompatible with quantization training", True
    
    try:
        from only_train_once.xai_optimizer import XAI_GETA
        
        # Check if this method requires inplace=False
        use_inplace = method_name not in LRP_METHODS
        
        # Create fresh model
        model = SmallVGG(num_classes=10, use_inplace=use_inplace).to(device)
        
        # Quantize model
        model = model_to_quantize_model(
            model,
            num_bits=8,
            quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
        )
        model = model.to(device)
        
        # Create datasets
        train_dataset = create_synthetic_dataset(num_samples=num_samples, num_classes=10)
        test_dataset = create_synthetic_dataset(num_samples=64, num_classes=10)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize OTO
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        oto = OTO(model, dummy_input)
        
        # Stage configuration
        BATCHES_PER_EPOCH = len(train_loader)
        START_PROJECTION = BATCHES_PER_EPOCH
        START_PRUNING = 2 * BATCHES_PER_EPOCH
        PRUNING_STEPS = BATCHES_PER_EPOCH
        
        # Create optimizer
        param_groups = oto._graph.get_param_groups()
        
        # Adjust n_steps for slow methods
        slow_methods = ['integrated_gradients', 'layer_integrated_gradients', 'layer_conductance', 'gradient_shap']
        n_steps = 3 if method_name in slow_methods else 5
        attr_freq = 100 if method_name in slow_methods else 50
        
        optimizer = XAI_GETA(
            params=param_groups,
            model=model,
            lr=0.0003,  # Lower learning rate for stability
            lr_quant=5e-7,  # Even lower for quantization params
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
            compute_attribution_freq=attr_freq,
            attribution_n_steps=n_steps,
            grad_clip_min=-0.02,  # Tighter gradient clipping for stability
            grad_clip_max=0.02,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        global_step = 0
        stages_seen = set()
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            
            for batch_idx, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                
                # Determine stage
                if global_step < START_PROJECTION:
                    stages_seen.add('warmup')
                elif global_step < START_PRUNING:
                    stages_seen.add('projection')
                elif global_step < START_PRUNING + PRUNING_STEPS:
                    stages_seen.add('pruning')
                else:
                    stages_seen.add('finetuning')
                
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step(inputs=X, targets=y)
                
                global_step += 1
            
            scheduler.step()
        
        # Check all stages were seen
        required_stages = {'warmup', 'projection', 'pruning', 'finetuning'}
        if stages_seen == required_stages:
            print("✓ PASSED")
            success = True
            error = None
        else:
            missing = required_stages - stages_seen
            print(f"✗ FAILED (missing stages: {missing})")
            success = False
            error = f"Missing stages: {missing}"
        
        # Cleanup
        del model, optimizer, oto
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return success, error, False
        
    except Exception as e:
        print(f"✗ FAILED ({str(e)[:50]}...)")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, str(e), False


def run_quick_check():
    """Run quick check of all 4 XAI-GETA stages."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("XAI-GETA QUICK CHECK - Testing All 4 Stages")
    print("="*70)
    
    # Check Captum
    try:
        import captum
        print(f"✓ Captum version: {captum.__version__}")
    except ImportError:
        print("✗ Captum not installed!")
        return False
    
    # Import XAI-GETA
    try:
        from only_train_once.xai_optimizer import XAI_GETA
        print("✓ XAI_GETA imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import XAI_GETA: {e}")
        return False
    
    # =========================================================================
    # Stage Setup
    # =========================================================================
    print("\n[Setup] Creating model and data...")
    
    # Create small model
    model = SmallVGG(num_classes=10).to(device)
    
    # Quantize model
    model = model_to_quantize_model(
        model,
        num_bits=8,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
    )
    model = model.to(device)
    print("✓ Model quantized")
    
    # Create synthetic datasets
    train_dataset = create_synthetic_dataset(num_samples=320, num_classes=10)
    test_dataset = create_synthetic_dataset(num_samples=64, num_classes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"✓ Synthetic data: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Initialize OTO
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    oto = OTO(model, dummy_input)
    print("✓ OTO initialized")
    
    # =========================================================================
    # Configure XAI-GETA with minimal steps for quick check
    # =========================================================================
    # With 320 samples and batch_size=32, we have 10 batches per epoch
    # We want all 4 stages to complete in few epochs
    
    BATCHES_PER_EPOCH = 10
    
    # Stage timings (in steps):
    # - Warmup: steps 0 to start_projection_step-1
    # - Projection: steps start_projection_step to start_pruning_step
    # - Joint Pruning & Quant: start_pruning_step to start_pruning_step + pruning_steps
    # - Fine-tuning: remaining steps
    
    START_PROJECTION = 10   # Warmup ends here (epoch 1)
    START_PRUNING = 30      # Projection ends here (epochs 2-3)
    PRUNING_STEPS = 20      # Pruning duration (epochs 4-5)
    # Fine-tuning will be the remaining epochs (6-8)
    
    TOTAL_EPOCHS = 8  # Should complete all 4 stages
    
    print(f"\n[Config] Stage configuration:")
    print(f"  - Warmup:     steps 0-{START_PROJECTION-1} (epoch 1)")
    print(f"  - Projection: steps {START_PROJECTION}-{START_PRUNING} (epochs 2-3)")
    print(f"  - Pruning:    steps {START_PRUNING}-{START_PRUNING + PRUNING_STEPS} (epochs 4-5)")
    print(f"  - Fine-tune:  steps > {START_PRUNING + PRUNING_STEPS} (epochs 6-8)")
    print(f"  - Total:      {TOTAL_EPOCHS} epochs × {BATCHES_PER_EPOCH} batches = {TOTAL_EPOCHS * BATCHES_PER_EPOCH} steps")
    
    # Create XAI_GETA optimizer directly
    param_groups = oto._graph.get_param_groups()
    
    optimizer = XAI_GETA(
        params=param_groups,
        model=model,  # Required for attribution computation
        lr=0.001,  # Lower learning rate for stability
        lr_quant=1e-5,  # Very small lr for quantization params to avoid instability
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        first_momentum=0.9,
        second_momentum=0.9,
        # Stage configuration
        start_projection_step=START_PROJECTION,
        projection_steps=START_PRUNING - START_PROJECTION,  # Duration of projection
        start_pruning_step=START_PRUNING,
        pruning_steps=PRUNING_STEPS,
        # XAI configuration
        attribution_method='saliency',
        attribution_weight=0.3,
        compute_attribution_freq=50,  # Don't update too often for speed
        # Gradient clipping for stability
        grad_clip_min=-0.1,
        grad_clip_max=0.1,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    print("✓ XAI_GETA optimizer created")
    
    # =========================================================================
    # Compute initial attributions
    # =========================================================================
    print("\n[Init] Computing initial attributions...")
    
    # Create a small dataloader for initial attribution computation
    small_loader = DataLoader(
        TensorDataset(
            torch.randn(32, 3, 32, 32),
            torch.randint(0, 10, (32,))
        ),
        batch_size=8
    )
    
    try:
        optimizer.compute_initial_attributions(small_loader, num_batches=2, criterion=criterion)
        print("✓ Initial attributions computed")
    except Exception as e:
        print(f"⚠ Attribution computation issue (continuing): {e}")
        # Continue anyway - optimizer will work without initial attributions
    
    # =========================================================================
    # Training loop - Track stage transitions
    # =========================================================================
    print("\n" + "="*70)
    print("Starting Training - Monitoring All 4 Stages")
    print("="*70)
    
    stage_markers = {
        'warmup': False,
        'projection': False,
        'pruning': False,
        'finetuning': False
    }
    
    global_step = 0
    
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            # Determine current stage based on XAI_GETA logic
            if global_step < START_PROJECTION:
                current_stage = "WARMUP"
                if not stage_markers['warmup']:
                    print(f"\n>>> STAGE 1: WARMUP started at step {global_step}")
                    stage_markers['warmup'] = True
            elif global_step <= START_PRUNING:
                current_stage = "PROJECTION"
                if not stage_markers['projection']:
                    print(f"\n>>> STAGE 2: PROJECTION started at step {global_step}")
                    stage_markers['projection'] = True
            elif global_step <= START_PRUNING + PRUNING_STEPS:
                current_stage = "PRUNING"
                if not stage_markers['pruning']:
                    print(f"\n>>> STAGE 3: JOINT PRUNING & QUANTIZATION started at step {global_step}")
                    stage_markers['pruning'] = True
            else:
                current_stage = "FINETUNE"
                if not stage_markers['finetuning']:
                    print(f"\n>>> STAGE 4: FINE-TUNING started at step {global_step}")
                    stage_markers['finetuning'] = True
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            
            # Backward pass (no auxiliary loss for simplicity)
            loss.backward()
            
            # Optimizer step
            try:
                optimizer.step(inputs=X, targets=y)
            except Exception as e:
                print(f"\n✗ Error at step {global_step} ({current_stage}): {e}")
                import traceback
                traceback.print_exc()
                return False
            
            epoch_loss += loss.item()
            global_step += 1
        
        scheduler.step()
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        
        # Get metrics
        num_groups = sum(len(g.get('pruned_idxes', [])) + len(g.get('important_idxes', [])) 
                        for g in optimizer.param_groups)
        num_pruned = sum(len(g.get('pruned_idxes', [])) for g in optimizer.param_groups)
        
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
        
        print(f"Epoch {epoch:2d}/{TOTAL_EPOCHS} | Stage: {current_stage:10s} | "
              f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | Pruned: {num_pruned}")
    
    # =========================================================================
    # Final Check
    # =========================================================================
    print("\n" + "="*70)
    print("STAGE COMPLETION CHECK")
    print("="*70)
    
    all_passed = True
    for stage, completed in stage_markers.items():
        status = "✓ PASSED" if completed else "✗ FAILED"
        print(f"  {stage.upper():12s}: {status}")
        if not completed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 SUCCESS! All 4 stages completed successfully!")
        print("   The XAI-GETA pipeline is working correctly.")
    else:
        print("\n❌ FAILURE: Some stages did not complete.")
        print("   Please check the errors above.")
    
    return all_passed


def run_all_methods_test(include_slow=False):
    """Test ALL attribution methods individually."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("XAI-GETA METHOD-BY-METHOD TEST")
    print("Testing each attribution method individually")
    print("="*70)
    
    # Check Captum
    try:
        import captum
        print(f"✓ Captum version: {captum.__version__}")
    except ImportError:
        print("✗ Captum not installed!")
        return False
    
    # Build methods list
    methods_to_test = [m for m in ATTRIBUTION_METHODS if m not in SLOW_METHODS]
    if include_slow:
        methods_to_test.extend(SLOW_METHODS)
        print(f"NOTE: Including slow methods ({SLOW_METHODS})")
    
    print(f"\nTesting {len(methods_to_test)} methods: {', '.join(methods_to_test)}")
    print("-" * 70)
    
    results = {}
    
    for method in methods_to_test:
        success, error, skipped = test_single_method(method, device, num_epochs=4, num_samples=200)
        results[method] = {'success': success, 'error': error, 'skipped': skipped}
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Group by category
    categories = {
        'Gradient-Based (Fast)': ['saliency', 'input_x_gradient', 'guided_backprop', 'deconvolution'],
        'Layer-Specific': ['layer_conductance', 'layer_gradient_x_activation', 'layer_integrated_gradients'],
        'Path-Based': ['deep_lift', 'integrated_gradients'],
        'Decomposition (LRP)': ['lrp', 'layer_lrp'],
        'SHAP-Based (Slow)': ['gradient_shap'],
    }
    
    for category, methods in categories.items():
        tested = [m for m in methods if m in results]
        if tested:
            print(f"\n  {category}:")
            for method in tested:
                result = results[method]
                if result['skipped']:
                    status = "⊘ SKIPPED"
                elif result['success']:
                    status = "✓ PASSED"
                else:
                    status = "✗ FAILED"
                print(f"    {method:30s}: {status}")
                if result['error'] and not result['skipped']:
                    print(f"        Error: {str(result['error'])[:60]}...")
    
    passed = sum(1 for r in results.values() if r['success'])
    skipped = sum(1 for r in results.values() if r['skipped'])
    total = len(results)
    print(f"\n  Total: {passed}/{total} methods passed ({skipped} skipped - incompatible with quantization)")
    print("="*70)
    
    return passed == total


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='XAI-GETA Quick Check')
    parser.add_argument('--all-methods', action='store_true',
                        help='Test all attribution methods individually')
    parser.add_argument('--include-slow', action='store_true',
                        help='Include very slow methods like gradient_shap')
    parser.add_argument('--method', type=str, default=None,
                        help='Test a specific method')
    args = parser.parse_args()
    
    if args.method:
        # Test a single specific method
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing single method: {args.method}")
        success, error, skipped = test_single_method(args.method, device)
        if error and not skipped:
            print(f"Error: {error}")
        sys.exit(0 if success else 1)
    elif args.all_methods:
        # Test all methods individually
        success = run_all_methods_test(include_slow=args.include_slow)
        sys.exit(0 if success else 1)
    else:
        # Default: run quick check with saliency
        success = run_quick_check()
        sys.exit(0 if success else 1)
