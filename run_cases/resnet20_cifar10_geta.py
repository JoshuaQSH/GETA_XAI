"""
ResNet20 CIFAR-10 GETA Experiment
=================================

Train ResNet20 on CIFAR-10 using the GETA optimizer for joint pruning and quantization.

Usage:
    python resnet20_cifar10_geta.py --epochs 200 --sparsity 0.7
    python resnet20_cifar10_geta.py --epochs 2  # Quick test

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
- Optimizer: SGD (1e-1) + StepLR
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
    evaluate_model,
    save_results_to_csv,
    print_results_summary,
    print_epoch_summary,
    print_config,
    get_timestamp,
    generate_experiment_name,
    ensure_output_dir,
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


def create_resnet20(device: str = 'cuda:0'):
    """Create non-quantized ResNet20 model for testing."""
    from sanity_check.backends.resnet20_cifar10 import resnet20_cifar10

    model = resnet20_cifar10()
    return model.to(device)


def train_geta(
    trainloader,
    testloader,
    model,
    oto,
    config: ExperimentConfig,
) -> ExperimentResults:
    """
    Train ResNet20 with GETA optimizer.

    Args:
        trainloader: Training data loader
        testloader: Test data loader
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

    # Create GETA optimizer
    param_groups = oto._graph.get_param_groups()

    optimizer = GETA(
        params=param_groups,
        variant="adam",  # Use adam like VGG7 to avoid quantization instability
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
        min_bit_act=config.min_bit,
        max_bit_act=config.max_bit,
        device=config.device,
    )
    
    # Store optimizer in OTO
    oto._optimizer = optimizer

    # Training setup
    # model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=0.1
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
    print(f"\nTraining ResNet20 with GETA for {config.epochs} epochs...")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Projection steps: {projection_steps}, Pruning steps: {pruning_steps}")
    print("-" * 80)

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Training
        # correct = 0
        # total = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{config.epochs}")
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

            # _, predicted = y_pred.max(1)
            # total += y.size(0)
            # correct += predicted.eq(y).sum().item()

            # pbar.set_postfix({
            #     'loss': f"{epoch_loss/(total//y.size(0)):.3f}",
            #     'acc': f"{100.*correct/total:.2f}%"
            # })
        
        # Update learning rate
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


def train_simple(trainloader, testloader, model, config):
    """Simple training function for testing."""
    import torch.optim as optim
    import torch.nn as nn

    device = config.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    best_acc = 0
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{config.epochs}, Batch {batch_idx}/{len(trainloader)}, Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.3f}%')

        # Testing
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        print(f'Epoch {epoch+1}/{config.epochs}: Test Loss: {test_loss/len(testloader):.3f}, Test Acc: {acc:.3f}%')

        if acc > best_acc:
            best_acc = acc

    return {'best_acc': best_acc, 'final_acc': acc}


def main():
    """Main function to run GETA experiment."""

    # Parse arguments
    parser = get_base_parser('ResNet20 CIFAR-10 GETA Experiment')
    args = parser.parse_args()
    config = args_to_config(args, is_xai=False)

    # Print configuration
    print("\n" + "=" * 70)
    print("ResNet20 CIFAR-10 GETA Experiment")
    print("=" * 70)
    print_config(config, 'GETA')

    # Ensure output directory exists
    ensure_output_dir(config)

    # Step 1: Create quantized ResNet20 model
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

    # Step 4: Train with GETA
    print("\n[Step 4] Training with GETA optimizer...")
    results, trained_model, oto = train_geta(
        trainloader, testloader, model, oto, config
    )

    # Print and save results
    print_results_summary(results)

    # Generate experiment name and save results
    exp_name = generate_experiment_name('geta', config, model_name='resnet20_cifar10')
    csv_path = os.path.join(config.output_dir, f"{exp_name}_results.csv")
    save_results_to_csv(results, csv_path)

    print("\n" + "=" * 70)
    print("GETA Experiment Complete!")
    print("=" * 70 + "\n")

    return results, trained_model, oto


if __name__ == "__main__":
    main()