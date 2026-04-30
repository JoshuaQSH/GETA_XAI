"""
XAI-GETA Demo: VGG7 on CIFAR-10 with Captum Attribution-based Importance Scoring

This demo showcases the XAI-enhanced GETA optimizer that uses Captum attribution
methods for more interpretable importance scoring during structured pruning.

Usage:
    python qvgg7_xaigeta_demo.py --method saliency --weight 0.3 --epochs 50
"""

import sys
import argparse
from tqdm import tqdm
import numpy as np

from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.vgg7 import vgg7_bn
from only_train_once import OTO
from only_train_once.xai_optimizer import XAI_GETA, CaptumAttributionCalculator
import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# Configuration
dataset_root = '../datasets'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Default values (can be overridden via command-line arguments)
ATTRMETHOD = 'saliency'
ATTRWEIGHT = 0.3    # Weight for attribution scores in importance calculation
MAX_EPOCH = 50


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='XAI-GETA Demo: VGG7 on CIFAR-10 with Captum Attribution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        default=ATTRMETHOD,
        choices=[
            'saliency',
            'input_x_gradient',
            'layer_conductance',
            'layer_gradient_x_activation',
            'layer_integrated_gradients',
            'deep_lift',
            'integrated_gradients',
            'lrp',
            'layer_lrp',
            'gradient_shap',
        ],
        help='Captum attribution method to use for importance scoring'
    )
    parser.add_argument(
        '--weight', '-w',
        type=float,
        default=ATTRWEIGHT,
        help='Weight for attribution scores in importance calculation (0.0-1.0)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=MAX_EPOCH,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=DEVICE,
        help='Device to use for training (e.g., cuda:0, cpu)'
    )
    return parser.parse_args()

# ============================================================================
# Helper Functions
# ============================================================================

def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def check_accuracy(model, testloader, device='cuda'):
    """Evaluate model accuracy on test set."""
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):
                X = batch['pixel_values'].to(device)
                y = batch['labels'].to(device)
            else:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
            
            y_pred = model(X)
            total += y.size(0)
            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            correct1 += prec1.item() * y.size(0)
            correct5 += prec5.item() * y.size(0)

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5


# ============================================================================
# Main Training Function
# ============================================================================

def train_xai_geta(trainloader, testloader, model, oto, trainset=None, max_epoch=10, 
                   attr_method='saliency', attr_weight=0.3, device='cuda:0'):
    """
    Train VGG7 with XAI-GETA optimizer.
    
    XAI-GETA uses Captum attribution methods for importance scoring,
    providing more interpretable pruning decisions.
    """
    
    # Get the underlying model for XAI_GETA
    underlying_model = model
    
    # Create XAI-GETA optimizer
    # Note: We manually create the optimizer to pass the model reference
    param_groups = oto._graph.get_param_groups()
    
    optimizer = XAI_GETA(
        params=param_groups,
        model=underlying_model,  # Pass model for Captum attribution
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0 * len(trainloader),
        projection_periods=5,
        projection_steps=10 * len(trainloader),  # 10 epochs (same as original GETA)
        start_pruning_step=10 * len(trainloader),
        pruning_periods=5,
        pruning_steps=10 * len(trainloader),  # 10 epochs (same as original GETA)
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=device,
        # XAI-specific parameters
        attribution_method=attr_method,  # Use saliency for lower memory usage
        attribution_weight=attr_weight,  # 30% weight (by default) for attribution scores
        ema_decay=0.9,
        compute_attribution_freq=100,  # Update attributions every 100 steps
        attribution_n_steps=5,  # Reduced for memory efficiency
    )
    
    # Store optimizer in OTO for later use
    oto._optimizer = optimizer
    
    print("\n" + "=" * 70)
    print("XAI-GETA Configuration:")
    print("=" * 70)
    print(f"Attribution Method: {optimizer.attribution_method}")
    print(f"Attribution Weight: {optimizer.attribution_weight}")
    print(f"Importance Score Criteria: {optimizer.importance_score_criteria}")
    print(f"Target Group Sparsity: {optimizer.target_group_sparsity}")
    print(f"Projection Steps: {optimizer.projection_steps}")
    print(f"Pruning Steps: {optimizer.pruning_steps}")
    print("=" * 70 + "\n")
    
    # Compute initial attributions during warmup (optional)
    print("Computing initial attributions (using smaller batches for memory efficiency)...")
    try:
        if trainset is not None:
            # Use smaller batch for initial attribution to save memory
            small_trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=16, shuffle=True, num_workers=2
            )
            optimizer.compute_initial_attributions(small_trainloader, num_batches=2)
            print(f"Initial attributions computed for {len(optimizer._cached_attributions)} layers")
            del small_trainloader
            torch.cuda.empty_cache()
        else:
            optimizer.compute_initial_attributions(trainloader, num_batches=2)
            print(f"Initial attributions computed for {len(optimizer._cached_attributions)} layers")
    except Exception as e:
        print(f"Warning: Could not compute initial attributions: {e}")
        print("Continuing with fallback importance scoring...")
    
    # Training setup
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # Training loop
    print("\nStarting XAI-GETA Training...")
    print("-" * 70)
    
    for epoch in range(max_epoch):
        f_avg_val = 0.0
        model.train()
        lr_scheduler.step()
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{max_epoch}")
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(device)
            y = y.to(device)
            
            # Forward pass
            y_pred = model.forward(X)
            f = criterion(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            f.backward()
            f_avg_val += f.item()
            
            # XAI-GETA step with input/target for attribution updates
            optimizer.step(inputs=X, targets=y)
            
            # Update progress bar
            pbar.set_postfix({'loss': f.item()})
        
        # Compute metrics
        opt_metrics = optimizer.compute_metrics()
        accuracy1, accuracy5 = check_accuracy(model, testloader, device=device)
        f_avg_val = f_avg_val / len(trainloader)
        
        print(f"\nEp: {epoch+1:2d} | Loss: {f_avg_val:.4f} | "
              f"Acc@1: {accuracy1:.2f}% | "
              f"GrpSparsity: {opt_metrics.group_sparsity:.4f} | "
              f"NormParams: {opt_metrics.norm_params:.2f} | "
              f"#Import: {opt_metrics.num_important_groups} | "
              f"#Redund: {opt_metrics.num_redundant_groups}")
    
    return model, oto


def main():
    """Main function to run the XAI-GETA demo."""
    
    print("\n" + "=" * 70)
    print("XAI-GETA Demo: VGG7 on CIFAR-10")
    print("Using Captum Attribution for Importance Scoring")
    print("=" * 70 + "\n")
    
    # Check Captum availability
    try:
        import captum
        print(f"[O] Captum version: {captum.__version__}")
    except ImportError:
        print(" [X] Captum not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "captum"])
        import captum
        print(f"[O] Captum installed: {captum.__version__}")
    
    # Step 1: Create model
    print("\n[Step 1] Creating VGG7 model with quantization...")
    model = vgg7_bn()
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    dummy_input = torch.rand(1, 3, 32, 32)
    
    # Step 2: Initialize OTO
    print("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model.to(DEVICE), dummy_input=dummy_input.to(DEVICE))
    
    # Step 3: Prepare datasets
    print("[Step 3] Preparing CIFAR-10 dataset...")
    trainset = CIFAR10(
        root=dataset_root, 
        train=True, 
        download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    testset = CIFAR10(
        root=dataset_root, 
        train=False, 
        download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=4
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    # Step 4: Train with XAI-GETA
    print("\n[Step 4] Training with XAI-GETA optimizer...")
    trained_model, oto = train_xai_geta(
        trainloader, 
        testloader, 
        model, 
        oto,
        trainset=trainset,  # Pass trainset for smaller batch attribution
        max_epoch=MAX_EPOCH,  # Run for 50 epochs as requested, map to the qvgg7_geta demo
        attr_method=ATTRMETHOD,
        attr_weight=ATTRWEIGHT,
        device=DEVICE
    )
    
    # Step 5: Final evaluation
    print("\n[Step 5] Final Evaluation...")
    final_acc1, final_acc5 = check_accuracy(trained_model, testloader, device=DEVICE)
    print(f"Final Top-1 Accuracy: {final_acc1:.2f}%")
    print(f"Final Top-5 Accuracy: {final_acc5:.2f}%")
    
    # Step 6: Show compression statistics
    print("\n[Step 6] Compression Statistics...")
    final_metrics = oto._optimizer.compute_metrics()
    print(f"Group Sparsity: {final_metrics.group_sparsity:.4f}")
    print(f"Total Parameter Norm: {final_metrics.norm_params:.2f}")
    print(f"Important Groups: {final_metrics.num_important_groups}")
    print(f"Redundant Groups: {final_metrics.num_redundant_groups}")
    print(f"Zero Groups: {final_metrics.num_zero_groups}")
    
    print("\n" + "=" * 70)
    print("XAI-GETA Demo Complete!")
    print("=" * 70 + "\n")
    
    return trained_model, oto


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Override global configuration with command-line arguments
    ATTRMETHOD = args.method
    ATTRWEIGHT = args.weight
    MAX_EPOCH = args.epochs
    DEVICE = args.device
    
    print(f"Using device: {args.device}")
    print(f"Attribution method: {ATTRMETHOD}")
    print(f"Attribution weight: {ATTRWEIGHT}")
    print(f"Max epochs: {MAX_EPOCH}")
    
    main()
