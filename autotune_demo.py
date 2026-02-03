#!/usr/bin/env python
"""
Autotuning Demo: VGG7-BN with sampled CIFAR-10

This script demonstrates the XAI hyperparameter autotuning tool
using a small dataset sample for quick iteration.

Usage:
    python autotune_demo.py --budget 10 --backend nevergrad
    python autotune_demo.py --budget 10 --backend optuna
"""

import argparse
import logging
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from only_train_once.autotune import (
    AutotuneXAI,
    XAISearchSpace,
    get_default_search_space,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # Test if GPU is accessible
            torch.cuda.get_device_properties(i)
            available_gpus.append(i)
        except Exception:
            continue
    
    return available_gpus


def run_trial_on_gpu(
    gpu_id: int,
    trial_ids: List[int],
    args: argparse.Namespace,
    data_dir: str,
    train_samples: int,
    test_samples: int,
    batch_size: int,
    search_space: XAISearchSpace,
    base_geta_config: Dict[str, Any]
) -> Tuple[int, Dict[str, Any]]:
    """
    Run a subset of trials on a specific GPU.
    
    Returns:
        Tuple of (gpu_id, results_dict)
    """
    # Set device for this process
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    logger.info(f"GPU {gpu_id}: Starting {len(trial_ids)} trials on {device}")
    
    # Create data loaders in this process
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        train_samples=train_samples,
        test_samples=test_samples,
        batch_size=batch_size,
        num_workers=0,  # Disable workers in subprocess
    )
    
    # Create autotuner for this GPU
    autotuner = AutotuneXAI(
        model_factory=lambda: VGG7_BN(num_classes=10),
        train_loader=train_loader,
        val_loader=test_loader,
        search_space=search_space,
        objective=args.objective,
        budget=len(trial_ids),  # Only run the assigned trials
        epochs_per_trial=args.epochs,
        optimizer_backend=args.backend,
        device=device,
        base_geta_config=base_geta_config,
        verbose=False,  # Reduce logging in parallel runs
    )
    
    # Run optimization
    best_config, best_score = autotuner.run()
    
    # Return results
    results = {
        'gpu_id': gpu_id,
        'trial_ids': trial_ids,
        'best_config': best_config,
        'best_score': best_score,
        'all_trials': autotuner.trial_results,
    }
    
    logger.info(f"GPU {gpu_id}: Completed {len(trial_ids)} trials, best score: {best_score:.4f}")
    
    return gpu_id, results


# =============================================================================
# Model Definition: VGG7-BN (Small VGG for CIFAR-10)
# =============================================================================

class VGG7_BN(nn.Module):
    """
    Small VGG-style network for CIFAR-10.
    
    Architecture:
        Conv(3,64) -> BN -> ReLU -> Conv(64,64) -> BN -> ReLU -> MaxPool
        Conv(64,128) -> BN -> ReLU -> Conv(128,128) -> BN -> ReLU -> MaxPool
        Conv(128,256) -> BN -> ReLU -> Conv(256,256) -> BN -> ReLU -> MaxPool
        FC(256*4*4, 256) -> ReLU -> FC(256, 10)
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# Data Loading
# =============================================================================

def get_cifar10_loaders(
    data_dir: str = '../datasets',
    train_samples: int = 5000,
    test_samples: int = 1000,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple:
    """
    Get CIFAR-10 data loaders with subsampling for quick experiments.
    
    Args:
        data_dir: Directory to store/load CIFAR-10
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
        batch_size: Batch size
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Subsample for quick experiments
    train_indices = list(range(min(train_samples, len(train_dataset))))
    test_indices = list(range(min(test_samples, len(test_dataset))))
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    logger.info(f"Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    
    return train_loader, test_loader


# =============================================================================
# Main
# =============================================================================

def main():
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set
        pass
    
    parser = argparse.ArgumentParser(description='XAI Autotuning Demo')
    parser.add_argument('--budget', type=int, default=10,
                        help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per trial')
    parser.add_argument('--backend', type=str, default='nevergrad',
                        choices=['nevergrad', 'optuna'],
                        help='Optimization backend')
    parser.add_argument('--train-samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--test-samples', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--objective', type=str, default='multi',
                        choices=['accuracy', 'smoothness', 'multi'],
                        help='Optimization objective')
    parser.add_argument('--output', type=str, default='autotune_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("XAI Hyperparameter Autotuning Demo")
    logger.info("=" * 60)
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Budget: {args.budget} trials")
    logger.info(f"Epochs per trial: {args.epochs}")
    logger.info(f"Objective: {args.objective}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load data
    logger.info("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        num_workers=0,  # Disable workers for multiprocessing compatibility
    )
    
    # Define search space (narrower for quick demo)
    search_space = XAISearchSpace(
        attribution_method=['saliency', 'layer_conductance'],  # Fast methods only
        attribution_weight=(0.2, 0.8),
        attribution_freq=(50, 200),
        attribution_n_steps=(5, 15),
        ema_decay=(0.8, 0.95),
        baseline_type=['zero', 'mean'],
    )
    
    # Base GETA config (not tuned)
    base_geta_config = {
        'variant': 'sgd',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'target_group_sparsity': 0.5,
        'start_pruning_step': 200,  # Start pruning after ~3 epochs on small dataset
    }
    
    # Multi-GPU setup
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.error("No GPUs available, cannot run autotuning")
        return None, 0.0
    
    num_gpus = args.num_gpus if args.num_gpus is not None else len(available_gpus)
    num_gpus = min(num_gpus, len(available_gpus))
    
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Using {num_gpus} GPUs for parallel autotuning")
    
    if num_gpus == 1:
        # Single GPU mode (original logic)
        logger.info("Running in single-GPU mode")
        autotuner = AutotuneXAI(
            model_factory=lambda: VGG7_BN(num_classes=10),
            train_loader=train_loader,
            val_loader=test_loader,
            search_space=search_space,
            objective=args.objective,
            budget=args.budget,
            epochs_per_trial=args.epochs,
            optimizer_backend=args.backend,
            device=args.device,
            base_geta_config=base_geta_config,
        )
        
        # Run optimization
        logger.info("\nStarting hyperparameter optimization...")
        start_time = time.time()
        
        best_config, best_score = autotuner.run()
        
        elapsed = time.time() - start_time
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed / 60:.1f} minutes")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best config:")
        if best_config is not None:
            for key, value in best_config.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("  No valid configuration found")
        
        # Save results
        autotuner.save_results(args.output)
        logger.info(f"\nResults saved to {args.output}")
        
        # Print YAML config
        if best_config is not None:
            logger.info("\nBest config as YAML:")
            logger.info(autotuner.get_best_config_yaml())
        
        return best_config, best_score
    
    else:
        # Multi-GPU parallel mode
        logger.info(f"Running in multi-GPU mode with {num_gpus} GPUs")
        
        # Distribute trials across GPUs
        trials_per_gpu = args.budget // num_gpus
        extra_trials = args.budget % num_gpus
        
        gpu_assignments = []
        trial_counter = 0
        
        for gpu_idx in range(num_gpus):
            gpu_id = available_gpus[gpu_idx]
            num_trials = trials_per_gpu + (1 if gpu_idx < extra_trials else 0)
            trial_ids = list(range(trial_counter, trial_counter + num_trials))
            trial_counter += num_trials
            
            gpu_assignments.append((gpu_id, trial_ids))
        
        logger.info("GPU assignments:")
        for gpu_id, trial_ids in gpu_assignments:
            logger.info(f"  GPU {gpu_id}: trials {trial_ids}")
        
        # Run parallel optimization
        logger.info("\nStarting parallel hyperparameter optimization...")
        start_time = time.time()
        
        # Prepare arguments for multiprocessing
        data_dir = '../datasets'  # Same as default in get_cifar10_loaders
        mp_args = [
            (gpu_id, trial_ids, args, data_dir, args.train_samples, args.test_samples, args.batch_size, search_space, base_geta_config)
            for gpu_id, trial_ids in gpu_assignments
        ]
        
        # Run in parallel
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.starmap(run_trial_on_gpu, mp_args)
        
        elapsed = time.time() - start_time
        
        # Consolidate results
        all_trial_results = []
        best_overall_score = -float('inf')
        best_overall_config = None
        
        for gpu_id, gpu_results in results:
            all_trial_results.extend(gpu_results['all_trials'])
            
            if gpu_results['best_score'] > best_overall_score:
                best_overall_score = gpu_results['best_score']
                best_overall_config = gpu_results['best_config']
        
        # Create consolidated results
        consolidated_results = {
            'optimization_results': {
                'best_config': best_overall_config,
                'best_score': best_overall_score,
                'total_trials': len(all_trial_results),
                'elapsed_time': elapsed,
            },
            'trial_results': all_trial_results,
            'multi_gpu_info': {
                'num_gpus_used': num_gpus,
                'gpu_assignments': gpu_assignments,
                'available_gpus': available_gpus,
            }
        }
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-GPU OPTIMIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed / 60:.1f} minutes")
        logger.info(f"GPUs used: {num_gpus}")
        logger.info(f"Total trials: {len(all_trial_results)}")
        logger.info(f"Best score: {best_overall_score:.4f}")
        logger.info(f"Best config:")
        if best_overall_config is not None:
            for key, value in best_overall_config.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("  No valid configuration found")
        
        # Save consolidated results
        import json
        with open(args.output, 'w') as f:
            json.dump(consolidated_results, f, indent=2, default=str)
        logger.info(f"\nConsolidated results saved to {args.output}")
        
        # Print YAML config
        if best_overall_config is not None:
            logger.info("\nBest config as YAML:")
            # Simple YAML-like output
            logger.info("xai_config:")
            for key, value in best_overall_config.items():
                logger.info(f"  {key}: {value}")
        
        return best_overall_config, best_overall_score


if __name__ == '__main__':
    main()
