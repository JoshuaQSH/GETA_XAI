"""
XAI-based Importance Score Calculation for GETA

This module provides importance score calculation using Captum attributions,
designed to work alongside or replace the original GETA importance scores.
"""

import torch
from typing import Dict, Optional, List
from only_train_once.transform import (
    TensorTransform,
    tensor_transformation_param_group,
)

from .captum_attribution import CaptumAttributionCalculator


def calculate_xai_importance_score(
    criteria: Dict[str, float],
    param_group: Dict,
    captum_calculator: Optional[CaptumAttributionCalculator] = None,
    cached_attributions: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Calculate importance scores using xAI attributions combined with traditional metrics.
    
    This function extends the original GETA importance score calculation by adding
    Captum attribution-based scoring.
    
    Args:
        criteria: Dict of criterion names to weights (e.g., {"magnitude": 0.2, "attribution": 0.3})
        param_group: Parameter group dictionary from optimizer
        captum_calculator: CaptumAttributionCalculator instance (optional)
        cached_attributions: Pre-computed attributions keyed by layer name (optional)
    """
    param_group['importance_scores'] = dict()
    
    with torch.no_grad():
        for cri_name in criteria:
            if cri_name == 'magnitude':
                _importance_score_by_magnitude(param_group)
            elif cri_name == 'avg_magnitude':
                _importance_score_by_avg_magnitude(param_group)
            elif cri_name == 'cosine_similarity':
                _importance_score_by_cosine_similarity(param_group)
            elif cri_name == 'taylor_first_order':
                _importance_score_by_first_order_taylor(param_group)
            elif cri_name == 'taylor_second_order':
                _importance_score_by_second_order_taylor(param_group)
            elif cri_name == 'attribution' or cri_name == 'captum_attribution':
                _importance_score_by_attribution(
                    param_group, 
                    captum_calculator, 
                    cached_attributions
                )
            elif cri_name == 'gradient_norm':
                _importance_score_by_gradient_norm(param_group)


def compute_attribution_importance(
    param_group: Dict,
    attribution_tensor: torch.Tensor,
    aggregation: str = "sum_abs",
) -> torch.Tensor:
    """
    Convert a layer attribution tensor to per-group importance scores.
    
    Args:
        param_group: Parameter group dictionary
        attribution_tensor: Attribution tensor from Captum
        aggregation: Aggregation method
        
    Returns:
        Per-group importance scores tensor
    """
    num_groups = param_group["num_groups"]
    device = param_group["params"][0].device if param_group["params"] else "cpu"
    
    # Handle different attribution shapes
    if attribution_tensor.dim() == 4:
        # Conv2d: [batch, channels, H, W]
        if aggregation == "sum_abs":
            scores = torch.abs(attribution_tensor).sum(dim=(0, 2, 3))
        elif aggregation == "mean_abs":
            scores = torch.abs(attribution_tensor).mean(dim=(0, 2, 3))
        else:
            scores = torch.abs(attribution_tensor).amax(dim=(0, 2, 3))
    elif attribution_tensor.dim() == 3:
        # Aggregated Conv2d attribution: [channels, H, W]
        if aggregation == "sum_abs":
            scores = torch.abs(attribution_tensor).sum(dim=(1, 2))
        elif aggregation == "mean_abs":
            scores = torch.abs(attribution_tensor).mean(dim=(1, 2))
        else:
            scores = torch.abs(attribution_tensor).amax(dim=(1, 2))
    elif attribution_tensor.dim() == 2:
        # Linear: [batch, features]
        if aggregation == "sum_abs":
            scores = torch.abs(attribution_tensor).sum(dim=0)
        elif aggregation == "mean_abs":
            scores = torch.abs(attribution_tensor).mean(dim=0)
        else:
            scores = torch.abs(attribution_tensor).amax(dim=0)
    elif attribution_tensor.dim() == 1:
        scores = torch.abs(attribution_tensor)
    else:
        # Flatten for other cases
        scores = torch.abs(attribution_tensor.flatten())
    
    # Adjust to num_groups
    if scores.numel() > num_groups:
        # Aggregate within groups
        group_size = scores.numel() // num_groups
        if group_size * num_groups <= scores.numel():
            scores = scores[:group_size * num_groups].view(num_groups, group_size).sum(dim=1)
        else:
            scores = scores[:num_groups]
    elif scores.numel() < num_groups:
        # Pad with zeros
        padded = torch.zeros(num_groups, device=scores.device)
        padded[:scores.numel()] = scores
        scores = padded
    
    return scores.to(device)


def _importance_score_by_attribution(
    param_group: Dict,
    captum_calculator: Optional[CaptumAttributionCalculator] = None,
    cached_attributions: Optional[Dict[str, torch.Tensor]] = None,
):
    """
    Compute importance scores using Captum attributions.
    
    Uses cached attributions if available, otherwise falls back to magnitude-based scoring.
    Attribution scores are normalized relative to magnitude scores for better calibration.
    """
    num_groups = param_group["num_groups"]
    device = param_group["params"][0].device if param_group["params"] else "cpu"
    
    # First compute magnitude-based scores for normalization reference
    mag_norm_group = None
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        if p_transform == TensorTransform.NO_PRUNE:
            continue
        param_transform = tensor_transformation_param_group(param.data, p_transform, param_group)
        if mag_norm_group is None:
            mag_norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            mag_norm_group += torch.norm(param_transform, dim=1) ** 2
    
    if mag_norm_group is not None:
        mag_scores = torch.sqrt(mag_norm_group)
    else:
        mag_scores = None
    
    # Try to get attribution from cache
    if cached_attributions is not None:
        layer_names = []
        for p_name in param_group["p_names"]:
            parts = p_name.rsplit('.', 1)
            if len(parts) == 2 and parts[1] in ['weight', 'bias']:
                layer_names.append(parts[0])

        unique_layer_names = list(dict.fromkeys(layer_names))
        layer_scores = []
        for layer_name in unique_layer_names:
            if layer_name in cached_attributions:
                attr_tensor = cached_attributions[layer_name]
                layer_scores.append(compute_attribution_importance(param_group, attr_tensor))

        if layer_scores:
            scores = torch.stack(layer_scores).mean(dim=0)

            if mag_scores is not None and scores.sum() > 1e-8:
                attr_scale = mag_scores.sum() / (scores.sum() + 1e-8)
                scores = scores * attr_scale

            param_group['importance_scores']['attribution'] = scores
            return
    
    # Fallback: use magnitude-based approximation
    # This ensures we always have a score even without attributions
    if mag_scores is not None:
        param_group['importance_scores']['attribution'] = mag_scores
    else:
        param_group['importance_scores']['attribution'] = torch.zeros(num_groups, device=device)


def _importance_score_by_magnitude(param_group: Dict):
    """Compute L2 norm based importance score per group."""
    norm_group = None
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        if p_transform == TensorTransform.NO_PRUNE:
            continue
        param_transform = tensor_transformation_param_group(param.data, p_transform, param_group)
        if norm_group is None:
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(param_transform, dim=1) ** 2
    
    if norm_group is not None:
        param_group['importance_scores']['magnitude'] = torch.sqrt(norm_group)
    else:
        num_groups = param_group["num_groups"]
        device = param_group["params"][0].device if param_group["params"] else "cpu"
        param_group['importance_scores']['magnitude'] = torch.zeros(num_groups, device=device)


def _importance_score_by_avg_magnitude(param_group: Dict):
    """Compute average magnitude per group."""
    norm_group = None
    group_sizes = 0
    for param, p_transform in zip(param_group['params'], param_group['p_transform']):
        if p_transform == TensorTransform.NO_PRUNE:
            continue
        param_transform = tensor_transformation_param_group(param.data, p_transform, param_group)
        if norm_group is None:
            norm_group = torch.norm(param_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(param_transform, dim=1) ** 2
        group_sizes += param_transform.shape[1] if len(param_transform.shape) > 1 else 1
    
    if norm_group is not None:
        param_group['importance_scores']['avg_magnitude'] = torch.sqrt(norm_group) / float(group_sizes + 1e-6)
    else:
        num_groups = param_group["num_groups"]
        device = param_group["params"][0].device if param_group["params"] else "cpu"
        param_group['importance_scores']['avg_magnitude'] = torch.zeros(num_groups, device=device)


def _importance_score_by_cosine_similarity(param_group: Dict):
    """Compute cosine similarity between parameters and gradients."""
    norm_params = None
    norm_grads = None
    params_grads_inner_prod = None
    
    for p_name, param, p_transform in zip(
        param_group['p_names'], param_group['params'], param_group['p_transform']
    ):
        if p_name not in param_group.get('grad_variant', {}):
            continue
        if p_transform == TensorTransform.NO_PRUNE:
            continue
            
        param_transform = tensor_transformation_param_group(param.data, p_transform, param_group)
        grad = param_group['grad_variant'][p_name]
        grad_transform = tensor_transformation_param_group(grad, p_transform, param_group)
        
        if norm_params is None:
            norm_params = torch.norm(param_transform, dim=1) ** 2
            norm_grads = torch.norm(grad_transform, dim=1) ** 2
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            norm_params += torch.norm(param_transform, dim=1) ** 2
            norm_grads += torch.norm(grad_transform, dim=1) ** 2
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)
    
    if norm_params is not None:
        norm_params = torch.sqrt(norm_params)
        norm_grads = torch.sqrt(norm_grads)
        eps = 1e-8
        param_group['importance_scores']['cosine_similarity'] = (
            params_grads_inner_prod / (norm_params + eps) / (norm_grads + eps) + 1
        )
    else:
        num_groups = param_group["num_groups"]
        device = param_group["params"][0].device if param_group["params"] else "cpu"
        param_group['importance_scores']['cosine_similarity'] = torch.ones(num_groups, device=device)


def _importance_score_by_first_order_taylor(param_group: Dict):
    """Compute first-order Taylor expansion: |w * grad|."""
    params_grads_inner_prod = None
    
    for p_name, param, p_transform in zip(
        param_group['p_names'], param_group['params'], param_group['p_transform']
    ):
        if p_name not in param_group.get('grad_variant', {}):
            continue
        if p_transform == TensorTransform.NO_PRUNE:
            continue
            
        param_transform = tensor_transformation_param_group(param.data, p_transform, param_group)
        grad = param_group['grad_variant'][p_name]
        grad_transform = tensor_transformation_param_group(grad, p_transform, param_group)
        
        if params_grads_inner_prod is None:
            params_grads_inner_prod = torch.sum(param_transform * grad_transform, dim=1)
        else:
            params_grads_inner_prod += torch.sum(param_transform * grad_transform, dim=1)
    
    if params_grads_inner_prod is not None:
        param_group['importance_scores']['taylor_first_order'] = torch.abs(params_grads_inner_prod)
    else:
        num_groups = param_group["num_groups"]
        device = param_group["params"][0].device if param_group["params"] else "cpu"
        param_group['importance_scores']['taylor_first_order'] = torch.zeros(num_groups, device=device)


def _importance_score_by_second_order_taylor(param_group: Dict):
    """Compute second-order Taylor approximation."""
    if 'taylor_first_order' in param_group['importance_scores']:
        param_group['importance_scores']['taylor_second_order'] = (
            0.5 * param_group['importance_scores']['taylor_first_order'] ** 2
        )
        return
    
    # Compute from scratch if first order not available
    _importance_score_by_first_order_taylor(param_group)
    if 'taylor_first_order' in param_group['importance_scores']:
        param_group['importance_scores']['taylor_second_order'] = (
            0.5 * param_group['importance_scores']['taylor_first_order'] ** 2
        )


def _importance_score_by_gradient_norm(param_group: Dict):
    """Compute gradient L2 norm per group."""
    norm_group = None
    
    for p_name, param, p_transform in zip(
        param_group['p_names'], param_group['params'], param_group['p_transform']
    ):
        if p_name not in param_group.get('grad_variant', {}):
            continue
        if p_transform == TensorTransform.NO_PRUNE:
            continue
            
        grad = param_group['grad_variant'][p_name]
        grad_transform = tensor_transformation_param_group(grad, p_transform, param_group)
        
        if norm_group is None:
            norm_group = torch.norm(grad_transform, dim=1) ** 2
        else:
            norm_group += torch.norm(grad_transform, dim=1) ** 2
    
    if norm_group is not None:
        param_group['importance_scores']['gradient_norm'] = torch.sqrt(norm_group)
    else:
        num_groups = param_group["num_groups"]
        device = param_group["params"][0].device if param_group["params"] else "cpu"
        param_group['importance_scores']['gradient_norm'] = torch.zeros(num_groups, device=device)
