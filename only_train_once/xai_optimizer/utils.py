"""
Utility functions for XAI-based optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def get_layer_names_for_pruning(model: nn.Module) -> List[str]:
    """
    Get layer names that are suitable for structured pruning.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of layer names
    """
    prunable_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            prunable_layers.append(name)
    return prunable_layers


def compute_layer_statistics(
    model: nn.Module,
    layer_name: str,
) -> Dict[str, float]:
    """
    Compute statistics for a specific layer.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            stats['weight_mean'] = param.data.mean().item()
            stats['weight_std'] = param.data.std().item()
            stats['weight_norm'] = param.data.norm().item()
            stats['num_params'] = param.numel()
            break
    return stats


def normalize_attributions(
    attributions: Dict[str, torch.Tensor],
    method: str = "minmax",
) -> Dict[str, torch.Tensor]:
    """
    Normalize attribution scores across layers.
    
    Args:
        attributions: Dictionary of layer attributions
        method: Normalization method ("minmax", "zscore", "l2")
        
    Returns:
        Normalized attributions
    """
    normalized = {}
    
    if method == "minmax":
        # Global min-max normalization
        all_values = torch.cat([attr.flatten() for attr in attributions.values()])
        global_min = all_values.min()
        global_max = all_values.max()
        scale = global_max - global_min + 1e-8
        
        for name, attr in attributions.items():
            normalized[name] = (attr - global_min) / scale
            
    elif method == "zscore":
        # Z-score normalization
        all_values = torch.cat([attr.flatten() for attr in attributions.values()])
        global_mean = all_values.mean()
        global_std = all_values.std() + 1e-8
        
        for name, attr in attributions.items():
            normalized[name] = (attr - global_mean) / global_std
            
    elif method == "l2":
        # L2 normalization
        for name, attr in attributions.items():
            norm = attr.norm() + 1e-8
            normalized[name] = attr / norm
    else:
        normalized = attributions
    
    return normalized


def aggregate_batch_attributions(
    batch_attributions: List[Dict[str, torch.Tensor]],
    method: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Aggregate attributions from multiple batches.
    
    Args:
        batch_attributions: List of attribution dictionaries
        method: Aggregation method ("mean", "max", "sum")
        
    Returns:
        Aggregated attributions
    """
    if not batch_attributions:
        return {}
    
    aggregated = {}
    layer_names = batch_attributions[0].keys()
    
    for name in layer_names:
        attrs = [batch[name] for batch in batch_attributions if name in batch]
        if not attrs:
            continue
            
        stacked = torch.stack(attrs)
        if method == "mean":
            aggregated[name] = stacked.mean(dim=0)
        elif method == "max":
            aggregated[name] = stacked.max(dim=0)[0]
        elif method == "sum":
            aggregated[name] = stacked.sum(dim=0)
        else:
            aggregated[name] = stacked.mean(dim=0)
    
    return aggregated


def print_attribution_summary(attributions: Dict[str, torch.Tensor]):
    """Print summary of attribution scores."""
    print("\n" + "=" * 60)
    print("Attribution Summary")
    print("=" * 60)
    
    for name, attr in attributions.items():
        print(f"\nLayer: {name}")
        print(f"  Shape: {attr.shape}")
        print(f"  Mean: {attr.mean().item():.6f}")
        print(f"  Std: {attr.std().item():.6f}")
        print(f"  Min: {attr.min().item():.6f}")
        print(f"  Max: {attr.max().item():.6f}")
    
    print("\n" + "=" * 60)
