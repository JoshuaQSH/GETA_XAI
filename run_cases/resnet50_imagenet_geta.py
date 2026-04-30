"""
ResNet50 ImageNet GETA Experiment
==================================

Train ResNet50 on ImageNet using the GETA optimizer for joint pruning and quantization.

Usage:
    python resnet50_imagenet_geta.py --epochs 90 --sparsity 0.7
    python resnet50_imagenet_geta.py --epochs 2  # Quick test

Experimental Setup:
- Bit width reduction: b_r = 2
- Bit width range: [b_l, b_u] = [4, 16]
- Exponential: t = 1
- Sparsity: 0.4, 0.5
- Total Epochs: 120
- Projection Periods B: 5
- Projection Steps K_b: 5 epochs
- Pruning Periods P: 10
- Pruning Steps K_p: 10 epochs
- Optimizer: SGD (1e-1) + StepLR
"""

import os
import sys
import time
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.optimizer import GETA

from run_cases.utils import (
    get_base_parser,
    args_to_config,
    ExperimentConfig,
    ExperimentResults,
    evaluate_model,
    save_results_to_csv,
    print_results_summary,
    print_epoch_summary,
    print_config,
    get_timestamp,
    generate_experiment_name,
    ensure_output_dir,
)


class RobustImageFolder(Dataset):
    """ImageFolder wrapper that handles corrupted images gracefully."""
    
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=None)
        self.transform = transform
        self.valid_indices = []
        self.corrupted_files = []
        
        # Pre-validate images (optional, can be slow for large datasets)
        self.valid_indices = list(range(len(self.dataset)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        path, target = self.dataset.samples[actual_idx]
        
        try:
            # Try to load the image
            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')  # Ensure RGB format
            
            if self.transform is not None:
                img = self.transform(img)
            
            return img, target
            
        except Exception as e:
            # Log the corrupted file
            if path not in self.corrupted_files:
                self.corrupted_files.append(path)
                print(f"Warning: Skipping corrupted image: {path} ({type(e).__name__})")
            
            # Return a random valid sample instead
            # Find another valid index
            for fallback_idx in range(len(self.valid_indices)):
                if fallback_idx != idx:
                    try:
                        fallback_actual = self.valid_indices[fallback_idx]
                        fallback_path, fallback_target = self.dataset.samples[fallback_actual]
                        with open(fallback_path, 'rb') as f:
                            img = Image.open(f)
                            img = img.convert('RGB')
                        if self.transform is not None:
                            img = self.transform(img)
                        return img, fallback_target
                    except:
                        continue
            
            # If all else fails, return a black image with label 0
            if self.transform is not None:
                # Create a dummy black image
                dummy = Image.new('RGB', (224, 224), (0, 0, 0))
                dummy = self.transform(dummy)
                return dummy, 0
            else:
                return Image.new('RGB', (224, 224), (0, 0, 0)), 0


def get_imagenet_loaders(config: ExperimentConfig) -> tuple:
    """Load ImageNet dataset and return train/test loaders."""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = ImageFolder(
        root=os.path.join(config.dataset_root, 'ImageNet', 'train'),
        transform=train_transform
    )

    testset = ImageFolder(
        root=os.path.join(config.dataset_root, 'ImageNet', 'val'),
        transform=test_transform
    )

    # Use RobustImageFolder to handle corrupted images
    # trainset = RobustImageFolder(
    #     root=os.path.join(config.dataset_root, 'ImageNet', 'train'),
    #     transform=train_transform
    # )

    # testset = RobustImageFolder(
    #     root=os.path.join(config.dataset_root, 'ImageNet', 'val'),
    #     transform=test_transform
    # )

    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return trainloader, testloader, trainset


def create_quantized_resnet50(device: str = 'cuda:0') -> tuple:
    """Create quantized ResNet50 model and dummy input."""
    import torchvision.models as models
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode

    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_ONLY,  # Only weight quantization supported by optimizer
        q_m_init=1.0,  # Use default q_m to avoid overflow
    )
    dummy_input = torch.rand(1, 3, 224, 224)

    return model.to(device), dummy_input.to(device)


def train_geta(
    trainloader,
    testloader,
    model,
    oto,
    config: ExperimentConfig,
) -> ExperimentResults:
    """
    Train ResNet50 with GETA optimizer.

    Args:
        trainloader: Training data loader
        testloader: Test data loader
        model: Quantized ResNet50 model
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
        variant="sgd",  # Use adam like VGG7 to avoid quantization instability
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
    print(f"\nTraining ResNet50 with GETA for {config.epochs} epochs...")
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
        model_name='ResNet50',
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

    return results


def main():
    """Main function to run GETA experiment."""

    # Parse arguments
    parser = get_base_parser('ResNet50 ImageNet GETA Experiment')
    args = parser.parse_args()
    config = args_to_config(args, is_xai=False)

    # Print configuration
    print("\n" + "=" * 70)
    print("ResNet50 ImageNet GETA Experiment")
    print("=" * 70)
    print_config(config, 'GETA')

    # Ensure output directory exists
    ensure_output_dir(config)

    # Step 1: Create model
    print("[Step 1] Creating quantized ResNet50 model...")
    model, dummy_input = create_quantized_resnet50(config.device)

    # Step 2: Initialize OTO
    print("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model, dummy_input=dummy_input)

    # Step 3: Load dataset
    print("[Step 3] Loading ImageNet dataset...")
    trainloader, testloader, trainset = get_imagenet_loaders(config)
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
    exp_name = generate_experiment_name('geta', config, model_name='resnet50_imagenet')
    csv_path = os.path.join(config.output_dir, f"{exp_name}_results.csv")
    save_results_to_csv(results, csv_path)

    print("\n" + "=" * 70)
    print("GETA Experiment Complete!")
    print("=" * 70 + "\n")

    return results, trained_model, oto


if __name__ == "__main__":
    main()