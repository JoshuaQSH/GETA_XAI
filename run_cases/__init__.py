"""
Run Cases Package
=================

Contains experiment scripts for GETA and XAI-GETA on VGG7/CIFAR-10.

Usage:
    # Run GETA experiment
    python -m run_cases.vgg7_cifar10_geta --epochs 200 --sparsity 0.7
    
    # Run XAI-GETA experiment
    python -m run_cases.vgg7_cifar10_xaigeta --method saliency --weight 0.3 --epochs 200
"""

from .utils import (
    ExperimentConfig,
    ExperimentResults,
    get_base_parser,
    get_xai_parser,
    args_to_config,
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
    AVAILABLE_METHODS,
)

__all__ = [
    'ExperimentConfig',
    'ExperimentResults',
    'get_base_parser',
    'get_xai_parser',
    'args_to_config',
    'get_cifar10_loaders',
    'create_quantized_vgg7',
    'evaluate_model',
    'save_results_to_csv',
    'print_results_summary',
    'print_epoch_summary',
    'print_config',
    'get_timestamp',
    'generate_experiment_name',
    'ensure_output_dir',
    'AVAILABLE_METHODS',
]
