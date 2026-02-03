"""
ResNet20 CIFAR-10 XAI-GETA Experiment
=====================================

Train ResNet20 on CIFAR-10 using the XAI-GETA optimizer with attribution-based importance scoring.

Usage:
    python resnet20_cifar10_xaigeta.py --method saliency --weight 0.3 --epochs 200
    python resnet20_cifar10_xaigeta.py --method integrated_gradients --weight 0.5 --epochs 2

Available Attribution Methods:
    - saliency
    - input_x_gradient
    - guided_backprop
    - deconvolution
    - layer_conductance
    - layer_gradient_x_activation
    - layer_integrated_gradients
    - deep_lift
    - integrated_gradients
    - lrp
    - layer_lrp

Experimental Setup:
- Bit width reduction: b_r = 2
- Bit width range: [b_l, b_u] = [4, 16]
- Exponential: t = 1
- Sparsity: 0.35
- Total Epochs: 350
- Projection Periods B: 7
- Projection Steps K_b: 35 epochs
- Pruning Periods P: 5
- Pruning Steps K_p: 30 epochs
- Optimizer: Adam (1e-3) + StepLR
- Attribution Methods: All Captum methods
"""

import os
import sys
import time
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.xai_optimizer import XAI_GETA, CaptumAttributionCalculator

from run_cases.utils import (
    get_xai_parser,
    args_to_config,
    ExperimentConfig,
    ExperimentResults,
    get_cifar10_loaders,
    evaluate_model,
    save_results_to_csv,
    print_results_summary,
    print_epoch_summary,
    print_config,
    get_timestamp,
    generate_experiment_name,
    ensure_output_dir,
    AVAILABLE_METHODS,
)


def create_quantized_resnet20(device: str = 'cuda:0') -> tuple:
    """Create quantized ResNet20 model and dummy input."""
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode

    model = resnet20_cifar10()
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
        q_m_init=1.0,  # Use default q_m to avoid overflow
    )
    dummy_input = torch.rand(1, 3, 32, 32)

    return model.to(device), dummy_input.to(device)


def train_xai_geta(
    trainloader,
    testloader,
    trainset,
    model,
    oto,
    config: ExperimentConfig,
) -> ExperimentResults:
    """
    Train ResNet20 with XAI-GETA optimizer.
    
    Args:
        trainloader: Training data loader
        testloader: Test data loader
        trainset: Training dataset (for attribution sampling)
        model: Quantized ResNet20 model
        oto: OTO instance
        config: Experiment configuration
        
    Returns:
        ExperimentResults with all metrics
    """
    
    # Calculate steps based on epochs
    steps_per_epoch = len(trainloader)
    projection_steps = config.projection_epochs * steps_per_epoch
    pruning_steps = config.pruning_epochs * steps_per_epoch
    start_pruning_step = projection_steps  # Start pruning after projection
    
    # Create XAI-GETA optimizer
    param_groups = oto._graph.get_param_groups()
    
    optimizer = XAI_GETA(
        params=param_groups,
        model=model,  # Pass model for Captum attribution
        variant="adam",
        lr=config.lr,
        lr_quant=config.lr_quant,
        first_momentum=0.9,
        weight_decay=config.weight_decay,
        target_group_sparsity=config.target_sparsity,
        start_projection_step=config.start_projection_step,
        projection_periods=config.projection_periods,
        projection_steps=projection_steps,
        start_pruning_step=start_pruning_step,
        pruning_periods=config.pruning_periods,
        pruning_steps=pruning_steps,
        bit_reduction=config.bit_reduction,
        min_bit_wt=config.min_bit,
        max_bit_wt=config.max_bit,
        device=config.device,
        # XAI-specific parameters
        attribution_method=config.attribution_method,
        attribution_weight=config.attribution_weight,
        ema_decay=config.ema_decay,
        compute_attribution_freq=config.update_freq,
        attribution_n_steps=config.attribution_n_steps,
    )
    
    # Store optimizer in OTO
    oto._optimizer = optimizer
    
    print("\n" + "=" * 70)
    print("XAI-GETA Optimizer Configuration:")
    print("=" * 70)
    print(f"Target Group Sparsity: {config.target_sparsity}")
    print(f"Projection Steps: {projection_steps} ({config.projection_epochs} epochs)")
    print(f"Pruning Steps: {pruning_steps} ({config.pruning_epochs} epochs)")
    print(f"Start Pruning Step: {start_pruning_step}")
    print(f"Bit Range: [{config.min_bit}, {config.max_bit}]")
    print(f"Bit Reduction: {config.bit_reduction}")
    print(f"Attribution Method: {config.attribution_method}")
    print(f"Attribution Weight: {config.attribution_weight}")
    print(f"Update Frequency: {config.update_freq}")
    print(f"Attribution N-Steps: {config.attribution_n_steps}")
    print("=" * 70 + "\n")
    
    # Compute initial attributions
    print("Computing initial attributions...")
    try:
        # Use smaller batch for initial attribution to save memory
        small_trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=16, shuffle=True, num_workers=2
        )
        optimizer.compute_initial_attributions(small_trainloader, num_batches=2)
        print(f"Initial attributions computed for {len(optimizer._cached_attributions)} layers")
        del small_trainloader
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Could not compute initial attributions: {e}")
        print("Continuing with fallback importance scoring...")
    
    # Training setup
    model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.lr_step_size, 
        gamma=config.lr_gamma
    )
    
    # Get initial MACs and BOPs
    print("Computing initial MACs and BOPs...")
    full_macs_info = oto.compute_macs(in_million=True)
    full_bops_info = oto.compute_bops(in_million=True)
    full_macs = full_macs_info["total"]
    full_bops = full_bops_info["total"]
    print(f"Full MACs: {full_macs:.2f} M")
    print(f"Full BOPs: {full_bops:.2f} M")
    
    # Training loop
    print("\nStarting XAI-GETA Training...")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(config.device), y.to(config.device)
            
            # Forward pass
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            
            # XAI-GETA step (includes attribution-based importance)
            optimizer.step()
            
            pbar.set_postfix({'loss': loss.item()})
        
        lr_scheduler.step()
        
        # Evaluate and print epoch summary
        avg_loss = epoch_loss / len(trainloader)
        acc1, acc5 = evaluate_model(model, testloader, config.device)
        metrics = optimizer.compute_metrics()
        
        print_epoch_summary(epoch, config.epochs, avg_loss, acc1, acc5, metrics)
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    final_acc1, final_acc5 = evaluate_model(model, testloader, config.device)
    final_metrics = optimizer.compute_metrics()
    
    # Compute compressed MACs and BOPs
    compressed_macs_info = oto.compute_macs(in_million=True)
    compressed_bops_info = oto.compute_bops(in_million=True)
    compressed_macs = compressed_macs_info["total"]
    compressed_bops = compressed_bops_info["total"]
    
    print(f"Compressed MACs: {compressed_macs:.2f} M")
    print(f"Compressed BOPs: {compressed_bops:.2f} M")
    
    # Create results
    results = ExperimentResults(
        model_name='ResNet20',
        optimizer_type='XAI-GETA',
        epochs=config.epochs,
        target_sparsity=config.target_sparsity,
        attribution_method=config.attribution_method,
        attribution_weight=config.attribution_weight,
        full_macs=full_macs,
        full_bops=full_bops,
        compressed_macs=compressed_macs,
        compressed_bops=compressed_bops,
        final_top1_accuracy=final_acc1,
        final_top5_accuracy=final_acc5,
        total_param_norm=final_metrics.norm_params,
        group_sparsity=final_metrics.group_sparsity,
        num_important_groups=final_metrics.num_important_groups,
        num_redundant_groups=final_metrics.num_redundant_groups,
        num_zero_groups=final_metrics.num_zero_groups,
        total_training_time=total_time,
        timestamp=get_timestamp(),
    )
    
    return results, model, oto


def main():
    """Main function to run XAI-GETA experiment."""
    
    # Parse arguments
    parser = get_xai_parser('ResNet20 CIFAR-10 XAI-GETA Experiment')
    args = parser.parse_args()
    config = args_to_config(args, is_xai=True)
    
    # Validate attribution method
    if config.attribution_method not in AVAILABLE_METHODS:
        print(f"Error: Unknown attribution method '{config.attribution_method}'")
        print(f"Available methods: {AVAILABLE_METHODS}")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("ResNet20 CIFAR-10 XAI-GETA Experiment")
    print("=" * 70)
    print_config(config, 'XAI-GETA')
    
    # Ensure output directory exists
    ensure_output_dir(config)
    
    # Step 1: Create model
    print("[Step 1] Creating quantized ResNet20 model...")
    model, dummy_input = create_quantized_resnet20(config.device)
    
    # Step 2: Initialize OTO
    print("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model, dummy_input=dummy_input)
    
    # Step 3: Load dataset
    print("[Step 3] Loading CIFAR-10 dataset...")
    trainloader, testloader, trainset = get_cifar10_loaders(config)
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testloader.dataset)}")
    
    # Step 4: Train with XAI-GETA
    print("\n[Step 4] Training with XAI-GETA optimizer...")
    print(f"Using attribution method: {config.attribution_method}")
    print(f"Attribution weight: {config.attribution_weight}")
    
    results, trained_model, oto = train_xai_geta(
        trainloader, testloader, trainset, model, oto, config
    )
    
    # Print and save results
    print_results_summary(results)
    
    # Generate experiment name and save results
    exp_name = generate_experiment_name('xai_geta', config, model_name='resnet20_cifar10')
    csv_path = os.path.join(config.output_dir, f"{exp_name}_results.csv")
    save_results_to_csv(results, csv_path)
    
    print("\n" + "=" * 70)
    print("XAI-GETA Experiment Complete!")
    print("=" * 70 + "\n")
    
    return results, trained_model, oto


if __name__ == "__main__":
    main()
