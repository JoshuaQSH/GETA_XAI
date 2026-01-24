"""
VGG7 CIFAR-10 GETA Experiment
=============================

Train VGG7 on CIFAR-10 using the GETA optimizer for joint pruning and quantization.

Usage:
    python vgg7_cifar10_geta.py --epochs 200 --sparsity 0.7
    python vgg7_cifar10_geta.py --epochs 10  # Quick test

Experimental Setup:
- Bit width reduction: b_r = 2
- Bit width range: [b_l, b_u] = [4, 16]
- Exponential: t = 1
- Sparsity: 0.7
- Total Epochs: 200
- Projection Periods B: 5
- Projection Steps K_b: 20 epochs
- Pruning Periods P: 10
- Pruning Steps K_p: 30 epochs
- Optimizer: Adam (1e-3) + StepLR
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
from only_train_once.optimizer import GETA

from run_cases.utils import (
    get_base_parser,
    args_to_config,
    ExperimentConfig,
    ExperimentResults,
    get_cifar10_loaders,
    create_quantized_vgg7,
    evaluate_model,
    save_results_to_csv,
    print_results_summary,
    print_epoch_summary,
    print_config,
    get_timestamp,
    generate_experiment_name,
    ensure_output_dir,
)


def train_geta(
    trainloader,
    testloader,
    model,
    oto,
    config: ExperimentConfig,
) -> ExperimentResults:
    """
    Train VGG7 with GETA optimizer.
    
    Args:
        trainloader: Training data loader
        testloader: Test data loader
        model: Quantized VGG7 model
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
    
    # Create GETA optimizer
    param_groups = oto._graph.get_param_groups()
    
    optimizer = GETA(
        params=param_groups,
        variant="adam",
        lr=config.lr,
        lr_quant=config.lr_quant,
        first_momentum=0.9,
        weight_decay=config.weight_decay,
        target_group_sparsity=config.target_sparsity,
        start_projection_step=0,
        projection_periods=config.projection_periods,
        projection_steps=projection_steps,
        start_pruning_step=start_pruning_step,
        pruning_periods=config.pruning_periods,
        pruning_steps=pruning_steps,
        bit_reduction=config.bit_reduction,
        min_bit_wt=config.min_bit,
        max_bit_wt=config.max_bit,
        device=config.device,
    )
    
    # Store optimizer in OTO
    oto._optimizer = optimizer
    
    print("\n" + "=" * 70)
    print("GETA Optimizer Configuration:")
    print("=" * 70)
    print(f"Target Group Sparsity: {config.target_sparsity}")
    print(f"Projection Steps: {projection_steps} ({config.projection_epochs} epochs)")
    print(f"Pruning Steps: {pruning_steps} ({config.pruning_epochs} epochs)")
    print(f"Start Pruning Step: {start_pruning_step}")
    print(f"Bit Range: [{config.min_bit}, {config.max_bit}]")
    print(f"Bit Reduction: {config.bit_reduction}")
    print("=" * 70 + "\n")
    
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
    dummy_input = torch.rand(1, 3, 32, 32).to(config.device)
    full_macs = oto.compute_macs(in_million=False)
    full_bops = oto.compute_bops(in_million=False)
    print(f"Full MACs: {full_macs / 1e6:.2f} M")
    print(f"Full BOPs: {full_bops / 1e6:.2f} M")
    
    # Training loop
    print("\nStarting GETA Training...")
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
            
            # GETA step
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
    compressed_macs = oto.compute_macs(in_million=False)
    compressed_bops = oto.compute_bops(in_million=False)
    
    print(f"Compressed MACs: {compressed_macs / 1e6:.2f} M")
    print(f"Compressed BOPs: {compressed_bops / 1e6:.2f} M")
    
    # Create results
    results = ExperimentResults(
        model_name='VGG7-BN',
        optimizer_type='GETA',
        epochs=config.epochs,
        target_sparsity=config.target_sparsity,
        attribution_method='',
        attribution_weight=0.0,
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
    """Main function to run GETA experiment."""
    
    # Parse arguments
    parser = get_base_parser('VGG7 CIFAR-10 GETA Experiment')
    args = parser.parse_args()
    config = args_to_config(args, is_xai=False)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("VGG7 CIFAR-10 GETA Experiment")
    print("=" * 70)
    print_config(config, 'GETA')
    
    # Ensure output directory exists
    ensure_output_dir(config)
    
    # Step 1: Create model
    print("[Step 1] Creating quantized VGG7 model...")
    model, dummy_input = create_quantized_vgg7(config.device)
    
    # Step 2: Initialize OTO
    print("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model, dummy_input=dummy_input)
    
    # Step 3: Load dataset
    print("[Step 3] Loading CIFAR-10 dataset...")
    trainloader, testloader, trainset = get_cifar10_loaders(config)
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testloader.dataset)}")
    
    # Step 4: Train with GETA
    print("\n[Step 4] Training with GETA optimizer...")
    results, trained_model, oto = train_geta(
        trainloader, testloader, model, oto, config
    )
    
    # Print and save results
    print_results_summary(results)
    
    # Generate experiment name and save results
    exp_name = generate_experiment_name('geta', config)
    csv_path = os.path.join(config.output_dir, f"{exp_name}_results.csv")
    save_results_to_csv(results, csv_path)
    
    print("\n" + "=" * 70)
    print("GETA Experiment Complete!")
    print("=" * 70 + "\n")
    
    return results, trained_model, oto


if __name__ == "__main__":
    main()
