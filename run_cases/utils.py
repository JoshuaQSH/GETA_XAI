"""
Shared Utilities for GETA and XAI-GETA Experiments
===================================================

This module provides common utilities for running GETA and XAI-GETA experiments,
including argument parsing, dataset loading, evaluation, and results logging.
"""

import os
import sys
import csv
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.vgg7 import vgg7_bn


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for GETA/XAI-GETA experiments."""
    # Dataset
    dataset_root: str = '../datasets'
    batch_size: int = 64
    num_workers: int = 4
    
    # Training
    epochs: int = 200
    lr: float = 1e-3
    lr_quant: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 50
    lr_gamma: float = 0.1
    
    # GETA/Quantization
    target_sparsity: float = 0.7
    bit_reduction: int = 2
    min_bit: int = 4
    max_bit: int = 16
    exponential_t: float = 1.0
    
    # Stage configuration (in epochs)
    start_projection_step: int = 0
    projection_periods: int = 5
    projection_epochs: int = 20
    pruning_periods: int = 10
    pruning_epochs: int = 30
    
    # XAI-GETA specific
    attribution_method: str = 'saliency'
    attribution_weight: float = 0.3
    attribution_freq: int = 100
    attribution_n_steps: int = 5
    ema_decay: float = 0.9
    num_samples: int = 10
    baseline_type: str = 'zero'
    update_freq: int = 100
    
    # Device
    device: str = 'cuda:0'
    
    # Output
    output_dir: str = './outputs'
    experiment_name: str = 'experiment'


# List of available attribution methods
AVAILABLE_METHODS = [
    'saliency', 'input_x_gradient', 'guided_backprop', 'deconvolution',
    'layer_conductance', 'layer_gradient_x_activation', 
    'layer_integrated_gradients', 'deep_lift', 'integrated_gradients',
    'lrp', 'layer_lrp', 'gradient_shap'
]


# ============================================================================
# YAML Config Loading
# ============================================================================

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict


def yaml_to_experiment_config(yaml_config: Dict[str, Any]) -> ExperimentConfig:
    """
    Convert a YAML config dictionary to an ExperimentConfig dataclass.
    
    Args:
        yaml_config: Dictionary loaded from YAML file
        
    Returns:
        ExperimentConfig instance
    """
    config = ExperimentConfig()
    
    # Dataset settings
    if 'dataset' in yaml_config:
        ds = yaml_config['dataset']
        config.dataset_root = ds.get('root', config.dataset_root)
        config.batch_size = ds.get('batch_size', config.batch_size)
        config.num_workers = ds.get('num_workers', config.num_workers)
    
    # Training settings
    if 'training' in yaml_config:
        tr = yaml_config['training']
        config.epochs = tr.get('epochs', config.epochs)
        config.lr = tr.get('lr', config.lr)
        config.lr_quant = tr.get('lr_quant', config.lr_quant)
        config.weight_decay = tr.get('weight_decay', config.weight_decay)
        config.lr_step_size = tr.get('lr_step_size', config.lr_step_size)
        config.lr_gamma = tr.get('lr_gamma', config.lr_gamma)
    
    # GETA/Quantization settings
    if 'geta' in yaml_config:
        geta = yaml_config['geta']
        config.target_sparsity = geta.get('target_sparsity', config.target_sparsity)
        config.bit_reduction = geta.get('bit_reduction', config.bit_reduction)
        config.min_bit = geta.get('min_bit', config.min_bit)
        config.max_bit = geta.get('max_bit', config.max_bit)
        config.exponential_t = geta.get('exponential_t', config.exponential_t)
    
    # Stage settings
    if 'stages' in yaml_config:
        stages = yaml_config['stages']
        config.start_projection_step = stages.get('start_projection_step', config.start_projection_step)
        config.projection_periods = stages.get('projection_periods', config.projection_periods)
        config.projection_epochs = stages.get('projection_epochs', config.projection_epochs)
        config.pruning_periods = stages.get('pruning_periods', config.pruning_periods)
        config.pruning_epochs = stages.get('pruning_epochs', config.pruning_epochs)
    
    # XAI settings
    if 'xai' in yaml_config:
        xai = yaml_config['xai']
        config.attribution_method = xai.get('attribution_method', config.attribution_method)
        config.attribution_weight = xai.get('attribution_weight', config.attribution_weight)
        config.attribution_freq = xai.get('attribution_freq', config.attribution_freq)
        config.attribution_n_steps = xai.get('attribution_n_steps', config.attribution_n_steps)
        config.ema_decay = xai.get('ema_decay', config.ema_decay)
        config.num_samples = xai.get('num_samples', config.num_samples)
        config.baseline_type = xai.get('baseline_type', config.baseline_type)
        config.update_freq = xai.get('update_freq', config.attribution_freq)  # Alias
    
    # Device
    config.device = yaml_config.get('device', config.device)
    
    # Output settings
    if 'output' in yaml_config:
        out = yaml_config['output']
        config.output_dir = out.get('dir', config.output_dir)
        config.experiment_name = out.get('experiment_name', config.experiment_name)
    
    return config


def merge_config_with_args(config: ExperimentConfig, args: argparse.Namespace, 
                           is_xai: bool = False) -> ExperimentConfig:
    """
    Merge ExperimentConfig with command-line arguments.
    CLI arguments override YAML config values (if explicitly provided).
    
    Args:
        config: Base ExperimentConfig (from YAML or defaults)
        args: Parsed command-line arguments
        is_xai: Whether this is an XAI-GETA experiment
        
    Returns:
        Merged ExperimentConfig
    """
    # Helper to check if arg was explicitly provided (not default)
    def arg_provided(arg_name: str, default_value: Any) -> bool:
        return hasattr(args, arg_name) and getattr(args, arg_name) != default_value
    
    # Dataset - always override if provided
    if hasattr(args, 'dataset_root') and args.dataset_root != '../datasets':
        config.dataset_root = args.dataset_root
    if hasattr(args, 'batch_size') and args.batch_size != 64:
        config.batch_size = args.batch_size
    if hasattr(args, 'num_workers') and args.num_workers != 4:
        config.num_workers = args.num_workers
    
    # Training
    if hasattr(args, 'epochs') and args.epochs != 200:
        config.epochs = args.epochs
    if hasattr(args, 'lr') and args.lr != 1e-3:
        config.lr = args.lr
    if hasattr(args, 'lr_quant') and args.lr_quant != 1e-3:
        config.lr_quant = args.lr_quant
    if hasattr(args, 'weight_decay') and args.weight_decay != 1e-4:
        config.weight_decay = args.weight_decay
    if hasattr(args, 'lr_step_size') and args.lr_step_size != 50:
        config.lr_step_size = args.lr_step_size
    if hasattr(args, 'lr_gamma') and args.lr_gamma != 0.1:
        config.lr_gamma = args.lr_gamma
    
    # GETA
    if hasattr(args, 'sparsity') and args.sparsity != 0.7:
        config.target_sparsity = args.sparsity
    if hasattr(args, 'bit_reduction') and args.bit_reduction != 2:
        config.bit_reduction = args.bit_reduction
    if hasattr(args, 'min_bit') and args.min_bit != 4:
        config.min_bit = args.min_bit
    if hasattr(args, 'max_bit') and args.max_bit != 16:
        config.max_bit = args.max_bit
    
    # Stages
    if hasattr(args, 'projection_periods') and args.projection_periods != 5:
        config.projection_periods = args.projection_periods
    if hasattr(args, 'projection_epochs') and args.projection_epochs != 20:
        config.projection_epochs = args.projection_epochs
    if hasattr(args, 'pruning_periods') and args.pruning_periods != 10:
        config.pruning_periods = args.pruning_periods
    if hasattr(args, 'pruning_epochs') and args.pruning_epochs != 30:
        config.pruning_epochs = args.pruning_epochs
    
    # Device
    if hasattr(args, 'device') and args.device != 'cuda:0':
        config.device = args.device
    
    # Output
    if hasattr(args, 'output_dir') and args.output_dir != './outputs':
        config.output_dir = args.output_dir
    if hasattr(args, 'experiment_name') and args.experiment_name != '':
        config.experiment_name = args.experiment_name
    
    # XAI-specific
    if is_xai:
        if hasattr(args, 'method') and args.method != 'saliency':
            config.attribution_method = args.method
        if hasattr(args, 'weight') and args.weight != 0.3:
            config.attribution_weight = args.weight
        if hasattr(args, 'attribution_freq') and args.attribution_freq != 100:
            config.attribution_freq = args.attribution_freq
        if hasattr(args, 'attribution_n_steps') and args.attribution_n_steps != 5:
            config.attribution_n_steps = args.attribution_n_steps
        if hasattr(args, 'ema_decay') and args.ema_decay != 0.9:
            config.ema_decay = args.ema_decay
        if hasattr(args, 'num_samples') and args.num_samples != 10:
            config.num_samples = args.num_samples
        if hasattr(args, 'baseline_type') and args.baseline_type != 'zero':
            config.baseline_type = args.baseline_type
        if hasattr(args, 'update_freq') and args.update_freq != 100:
            config.update_freq = args.update_freq
    
    return config


@dataclass
class ExperimentResults:
    """Results from a GETA/XAI-GETA experiment."""
    # Model info
    model_name: str = ''
    optimizer_type: str = ''  # 'geta' or 'xai_geta'
    
    # Configuration
    epochs: int = 0
    target_sparsity: float = 0.0
    attribution_method: str = ''
    attribution_weight: float = 0.0
    
    # MACs and BOPs
    full_macs: float = 0.0
    full_bops: float = 0.0
    compressed_macs: float = 0.0
    compressed_bops: float = 0.0
    
    # Accuracy
    final_top1_accuracy: float = 0.0
    final_top5_accuracy: float = 0.0
    
    # Compression metrics
    total_param_norm: float = 0.0
    group_sparsity: float = 0.0
    num_important_groups: int = 0
    num_redundant_groups: int = 0
    num_zero_groups: int = 0
    
    # Timing
    total_training_time: float = 0.0
    timestamp: str = ''


# ============================================================================
# Argument Parsing
# ============================================================================

def get_base_parser(description: str = 'GETA Experiment') -> argparse.ArgumentParser:
    """Get base argument parser with common arguments."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (highest priority after CLI args)
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to YAML configuration file')
    
    # Dataset
    parser.add_argument('--dataset-root', type=str, default='../datasets',
                        help='Root directory for datasets')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Training
    parser.add_argument('--epochs', '-e', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lr-quant', type=float, default=1e-3,
                        help='Learning rate for quantization parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr-step-size', type=int, default=50,
                        help='StepLR step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='StepLR gamma')
    
    # GETA/Quantization
    parser.add_argument('--sparsity', type=float, default=0.7,
                        help='Target group sparsity')
    parser.add_argument('--bit-reduction', type=int, default=2,
                        help='Bit width reduction')
    parser.add_argument('--min-bit', type=int, default=4,
                        help='Minimum bit width')
    parser.add_argument('--max-bit', type=int, default=16,
                        help='Maximum bit width')
    
    # Stage configuration
    parser.add_argument('--start-projection-step', type=int, default=0, 
                        help='Starting step for projection')
    parser.add_argument('--projection-periods', type=int, default=5,
                        help='Number of projection periods')
    parser.add_argument('--projection-epochs', type=int, default=20,
                        help='Number of projection epochs')
    parser.add_argument('--pruning-periods', type=int, default=10,
                        help='Number of pruning periods')
    parser.add_argument('--pruning-epochs', type=int, default=30,
                        help='Number of pruning epochs')
    
    # Device
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory for output files')
    parser.add_argument('--experiment-name', type=str, default='',
                        help='Name for this experiment (auto-generated if empty)')
    
    return parser


def get_xai_parser(description: str = 'XAI-GETA Experiment') -> argparse.ArgumentParser:
    """Get argument parser with XAI-GETA specific arguments."""
    parser = get_base_parser(description)
    
    # XAI-GETA specific
    parser.add_argument('--method', '-m', type=str, default='saliency',
                        choices=[
                            'saliency', 'input_x_gradient', 'guided_backprop', 'deconvolution',
                            'layer_conductance', 'layer_gradient_x_activation', 
                            'layer_integrated_gradients', 'deep_lift', 'integrated_gradients',
                            'lrp', 'layer_lrp', 'gradient_shap'
                        ],
                        help='Attribution method for importance scoring')
    parser.add_argument('--weight', '-w', type=float, default=0.3,
                        help='Weight for attribution scores (0.0-1.0)')
    parser.add_argument('--attribution-freq', type=int, default=100,
                        help='Frequency of attribution updates (steps)')
    parser.add_argument('--attribution-n-steps', type=int, default=5,
                        help='Number of steps for integrated gradients methods')
    parser.add_argument('--ema-decay', type=float, default=0.9,
                        help='EMA decay for attribution smoothing')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for attribution computation')
    parser.add_argument('--baseline-type', type=str, default='zero',
                        choices=['zero', 'random', 'mean'],
                        help='Baseline type for attribution methods')
    parser.add_argument('--update-freq', type=int, default=100,
                        help='Frequency of attribution updates (steps)')
    
    return parser


def args_to_config(args, is_xai: bool = False) -> ExperimentConfig:
    """
    Convert parsed arguments to ExperimentConfig.
    
    Priority: CLI args > YAML config > defaults
    
    If --config is provided, load from YAML first, then override with any 
    explicitly provided CLI arguments.
    """
    # Start with defaults or load from YAML
    if hasattr(args, 'config') and args.config is not None:
        print(f"Loading configuration from: {args.config}")
        yaml_config = load_config_from_yaml(args.config)
        config = yaml_to_experiment_config(yaml_config)
        # Merge CLI args (they override YAML values)
        config = merge_config_with_args(config, args, is_xai=is_xai)
    else:
        # No YAML config, use CLI args directly
        config = ExperimentConfig(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            lr_quant=args.lr_quant,
            weight_decay=args.weight_decay,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            target_sparsity=args.sparsity,
            bit_reduction=args.bit_reduction,
            min_bit=args.min_bit,
            max_bit=args.max_bit,
            projection_periods=args.projection_periods,
            projection_epochs=args.projection_epochs,
            pruning_periods=args.pruning_periods,
            pruning_epochs=args.pruning_epochs,
            device=args.device,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
        )
        
        if is_xai:
            config.attribution_method = args.method
            config.attribution_weight = args.weight
            config.attribution_freq = args.attribution_freq
            config.attribution_n_steps = args.attribution_n_steps
            config.ema_decay = args.ema_decay
            config.num_samples = args.num_samples
            config.baseline_type = args.baseline_type
            config.update_freq = args.update_freq
    
    return config


# ============================================================================
# Dataset Loading
# ============================================================================

def get_cifar10_loaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader, CIFAR10]:
    """Load CIFAR-10 dataset and return train/test loaders."""
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = CIFAR10(
        root=config.dataset_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    testset = CIFAR10(
        root=config.dataset_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
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


# ============================================================================
# Model Creation
# ============================================================================

def create_quantized_vgg7(device: str = 'cuda:0') -> Tuple[nn.Module, torch.Tensor]:
    """Create quantized VGG7 model and dummy input."""
    model = vgg7_bn()
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
    )
    dummy_input = torch.rand(1, 3, 32, 32)
    
    return model.to(device), dummy_input.to(device)


# ============================================================================
# Evaluation
# ============================================================================

def accuracy_topk(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    """Compute precision@k for specified values of k."""
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


def evaluate_model(model: nn.Module, testloader: DataLoader, device: str = 'cuda') -> Tuple[float, float]:
    """Evaluate model accuracy on test set."""
    model.eval()
    correct1 = 0
    correct5 = 0
    total = 0
    
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            total += y.size(0)
            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            correct1 += prec1.item() * y.size(0)
            correct5 += prec5.item() * y.size(0)

    model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5


# ============================================================================
# Results Logging
# ============================================================================

def save_results_to_csv(results: ExperimentResults, filepath: str):
    """Save experiment results to CSV file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results).keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(results))
    
    print(f"Results saved to: {filepath}")


def print_results_summary(results: ExperimentResults):
    """Print a formatted summary of experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model: {results.model_name}")
    print(f"Optimizer: {results.optimizer_type}")
    print(f"Epochs: {results.epochs}")
    print(f"Target Sparsity: {results.target_sparsity}")
    if results.attribution_method:
        print(f"Attribution Method: {results.attribution_method}")
        print(f"Attribution Weight: {results.attribution_weight}")
    print("-" * 70)
    # Values are already in millions from compute_macs/compute_bops
    print(f"Full MACs:       {results.full_macs:.2f} M")
    print(f"Full BOPs:       {results.full_bops:.2f} M")
    print(f"Compressed MACs: {results.compressed_macs:.2f} M")
    print(f"Compressed BOPs: {results.compressed_bops:.2f} M")
    if results.full_macs > 0:
        print(f"MACs Reduction:  {(1 - results.compressed_macs / results.full_macs) * 100:.2f}%")
    if results.full_bops > 0:
        print(f"BOPs Reduction:  {(1 - results.compressed_bops / results.full_bops) * 100:.2f}%")
    print("-" * 70)
    print(f"Final Top-1 Accuracy: {results.final_top1_accuracy:.2f}%")
    print(f"Final Top-5 Accuracy: {results.final_top5_accuracy:.2f}%")
    print("-" * 70)
    print(f"Group Sparsity:    {results.group_sparsity:.4f}")
    print(f"Param Norm:        {results.total_param_norm:.2f}")
    print(f"Important Groups:  {results.num_important_groups}")
    print(f"Redundant Groups:  {results.num_redundant_groups}")
    print(f"Zero Groups:       {results.num_zero_groups}")
    print("-" * 70)
    print(f"Training Time: {results.total_training_time:.2f}s ({results.total_training_time/60:.2f}min)")
    print(f"Timestamp: {results.timestamp}")
    print("=" * 70 + "\n")


def print_epoch_summary(epoch: int, max_epoch: int, loss: float, 
                        acc1: float, acc5: float, metrics: Any):
    """Print epoch training summary."""
    print(f"Ep: {epoch:3d}/{max_epoch} | "
          f"Loss: {loss:.4f} | "
          f"Acc@1: {acc1:.2f}% | "
          f"Acc@5: {acc5:.2f}% | "
          f"GrpSparsity: {metrics.group_sparsity:.4f} | "
          f"NormParams: {metrics.norm_params:.2f} | "
          f"#Imp: {metrics.num_important_groups} | "
          f"#Red: {metrics.num_redundant_groups}")


def print_config(config: ExperimentConfig, optimizer_type: str = 'GETA'):
    """Print experiment configuration."""
    print("\n" + "=" * 70)
    print(f"{optimizer_type} Experiment Configuration")
    print("=" * 70)
    print(f"Dataset Root: {config.dataset_root}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.lr}")
    print(f"LR Quantization: {config.lr_quant}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"LR Step Size: {config.lr_step_size}")
    print(f"LR Gamma: {config.lr_gamma}")
    print("-" * 70)
    print(f"Target Sparsity: {config.target_sparsity}")
    print(f"Bit Reduction: {config.bit_reduction}")
    print(f"Bit Range: [{config.min_bit}, {config.max_bit}]")
    print(f"Projection Periods: {config.projection_periods}")
    print(f"Projection Epochs: {config.projection_epochs}")
    print(f"Pruning Periods: {config.pruning_periods}")
    print(f"Pruning Epochs: {config.pruning_epochs}")
    if optimizer_type == 'XAI-GETA':
        print("-" * 70)
        print(f"Attribution Method: {config.attribution_method}")
        print(f"Attribution Weight: {config.attribution_weight}")
        print(f"Attribution Freq: {config.attribution_freq}")
        print(f"Attribution N-Steps: {config.attribution_n_steps}")
        print(f"EMA Decay: {config.ema_decay}")
    print("-" * 70)
    print(f"Device: {config.device}")
    print(f"Output Dir: {config.output_dir}")
    print("=" * 70 + "\n")


# ============================================================================
# Utility Functions
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_experiment_name(optimizer_type: str, config: ExperimentConfig) -> str:
    """Generate experiment name from configuration."""
    if config.experiment_name:
        return config.experiment_name
    
    name_parts = [
        f"vgg7_cifar10_{optimizer_type.lower()}",
        f"sp{config.target_sparsity}",
        f"ep{config.epochs}"
    ]
    
    if optimizer_type.lower() == 'xai_geta':
        name_parts.append(f"{config.attribution_method}")
        name_parts.append(f"w{config.attribution_weight}")
    
    return "_".join(name_parts)


def ensure_output_dir(config: ExperimentConfig):
    """Ensure output directory exists."""
    os.makedirs(config.output_dir, exist_ok=True)
