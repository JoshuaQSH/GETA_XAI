"""
XAI-Enhanced GETA Optimizer

This module provides the XAI_GETA optimizer that extends the original GETA
with Captum attribution-based importance scoring.
"""

import logging
import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import required

from only_train_once.transform import (
    TensorTransform,
    tensor_transformation_param_group,
)
from only_train_once.optimizer.base_hybrid_sparse_optimizer import BaseHybridSparseOptimizer

from .captum_attribution import CaptumAttributionCalculator, AttributionMethod, CAPTUM_AVAILABLE
from .xai_importance_score import calculate_xai_importance_score


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class XAI_GETA(BaseHybridSparseOptimizer):
    """
    XAI-Enhanced GETA: Explainable AI-guided Training framework for 
    joint structured pruning and quantization.
    
    This optimizer extends GETA by incorporating Captum attribution methods
    for more interpretable and effective importance scoring.
    """

    def __init__(
        self,
        params,
        model: nn.Module = None,
        variant="sgd",
        lr=required,
        lr_quant=1e-3,
        first_momentum=None,
        second_momentum=None,
        dampening=None,
        weight_decay=None,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_steps=1,
        projection_periods=1,
        start_pruning_step=1,
        pruning_steps=1,
        pruning_periods=1,
        group_divisible=1,
        importance_score_criteria="default",
        bit_reduction=2,
        min_bit_wt=2,
        max_bit_wt=16,
        min_bit_act=2,
        max_bit_act=16,
        grad_clip_min=-1.0,
        grad_clip_max=1.0,
        verbose="False",
        device="cuda",
        log_dir="outputs",
        # XAI-specific parameters
        attribution_method: str = "layer_conductance",
        attribution_weight: float = 0.3,
        ema_decay: float = 0.9,
        attribution_n_steps: int = 20,
        compute_attribution_freq: int = 100,
    ):
        """
        Initialize XAI_GETA optimizer.
        
        Args:
            params: Model parameters
            model: The neural network model (required for attribution computation)
            variant: Optimizer variant (sgd, adam, adamw)
            lr: Learning rate for model parameters
            lr_quant: Learning rate for quantization parameters
            ... (same as GETA)
            attribution_method: Captum attribution method to use
            attribution_weight: Weight for attribution in importance score
            ema_decay: EMA decay for attribution score updates
            attribution_n_steps: Number of steps for integrated gradients methods
            compute_attribution_freq: How often to recompute attributions (steps)
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # GETA parameters
        self.start_projection_step = start_projection_step
        self.projection_steps = projection_steps
        self.projection_periods = projection_periods
        self.projection_period_duration = (
            self.projection_steps // self.projection_periods if self.projection_periods > 0 else 1
        )
        self.start_pruning_step = start_pruning_step
        self.pruning_periods = int(max(1, pruning_periods))
        self.pruning_steps = pruning_steps
        self.pruning_period_duration = (
            self.pruning_steps // self.pruning_periods if self.pruning_periods > 0 else 1
        )
        self.curr_pruning_period = 0
        self.lr_quant = lr_quant
        self.bit_reduction = bit_reduction
        self.min_bit_wt = min_bit_wt
        self.max_bit_wt = max_bit_wt
        self.min_bit_act = min_bit_act
        self.max_bit_act = max_bit_act
        self.grad_clip_min = grad_clip_min
        self.grad_clip_max = grad_clip_max
        self.verbose = verbose
        self.device = device
        self.pruned_group_idxes = list()
        self.gamma = 0.0
        self.d_quant = 0.0
        self.bit_layers = {}
        
        # XAI-specific parameters
        self.attribution_method = attribution_method
        self.attribution_weight = attribution_weight
        self.ema_decay = ema_decay
        self.compute_attribution_freq = compute_attribution_freq
        self._model = model
        self._last_attribution_step = -1
        
        # Cached attribution scores
        self._cached_attributions: Dict[str, torch.Tensor] = {}
        self._initial_attributions_computed = False
        
        # Initialize Captum calculator if model is provided
        self.captum_calculator = None
        if model is not None and CAPTUM_AVAILABLE:
            self.captum_calculator = CaptumAttributionCalculator(
                model=model,
                attribution_method=attribution_method,
                n_steps=attribution_n_steps,
                device=device,
            )
            self.logger.info(f"Initialized Captum calculator with method: {attribution_method}")
        elif not CAPTUM_AVAILABLE:
            self.logger.warning("Captum not available. Falling back to traditional importance scoring.")
        
        # Set up importance score criteria
        if importance_score_criteria == "default":
            if self.captum_calculator is not None:
                # Use attribution as part of the criteria
                self.importance_score_criteria = {
                    "attribution": attribution_weight,
                    "magnitude": (1.0 - attribution_weight) * 0.3,
                    "taylor_first_order": (1.0 - attribution_weight) * 0.4,
                    "cosine_similarity": (1.0 - attribution_weight) * 0.3,
                }
            else:
                # Fallback to original GETA criteria
                self.importance_score_criteria = {
                    "magnitude": 0.2,
                    "avg_magnitude": 0.2,
                    "cosine_similarity": 0.2,
                    "taylor_first_order": 0.2,
                    "taylor_second_order": 0.2,
                }
        else:
            self.importance_score_criteria = importance_score_criteria
        
        additional_defaults = dict(
            lr_quant=lr_quant,
        )

        super().__init__(
            params,
            variant=variant,
            lr=lr,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            target_group_sparsity=target_group_sparsity,
            group_divisible=group_divisible,
            additional_defaults=additional_defaults,
        )
        
        # Initialize param group attributes (same as GETA)
        for param_group in self.param_groups:
            param_group["important_idxes"] = [
                i for i in range(param_group["num_groups"])
            ]
            param_group["active_redundant_idxes"] = list()
            param_group["pruned_idxes"] = list()
            param_group["importance_scores"] = dict()
            param_group["lr_quant"] = lr_quant
        
        # Set up active number redundant groups for each pruning period
        self.active_num_redundant_groups = list()
        groups_sum = 0
        for p in range(self.pruning_periods):
            if p == self.pruning_periods - 1:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups - groups_sum
                )
            else:
                self.active_num_redundant_groups.append(
                    self.target_num_redundant_groups // self.pruning_periods
                )
                groups_sum += self.target_num_redundant_groups // self.pruning_periods
        
        self.logger.info(f"XAI_GETA initialized with criteria: {self.importance_score_criteria}")

    def compute_initial_attributions(
        self,
        dataloader,
        num_batches: int = 5,
        criterion=None,
    ):
        """
        Compute initial attribution scores during warmup phase.
        
        This should be called at the beginning of training to establish
        baseline attribution scores.
        
        Args:
            dataloader: Training data loader
            num_batches: Number of batches to use for attribution computation
            criterion: Loss criterion (optional, for gradient computation)
        """
        if self.captum_calculator is None:
            self.logger.warning("Captum calculator not available. Skipping initial attributions.")
            return
        
        self.logger.info("Computing initial attribution scores...")
        self._model.eval()
        
        attribution_accumulator: Dict[str, List[torch.Tensor]] = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                if isinstance(batch, dict):
                    X = batch['pixel_values'].to(self.device)
                    y = batch['labels'].to(self.device)
                else:
                    X, y = batch
                    X = X.to(self.device)
                    y = y.to(self.device)
                
                # Compute attributions for this batch
                attributions = self.captum_calculator.compute_all_layer_attributions(X, y)
                
                for layer_name, attr in attributions.items():
                    # Aggregate along batch dimension to get consistent shape
                    if attr.dim() >= 2:
                        attr_aggregated = attr.abs().sum(dim=0) / attr.shape[0]
                    else:
                        attr_aggregated = attr.abs()
                    
                    if layer_name not in attribution_accumulator:
                        attribution_accumulator[layer_name] = []
                    attribution_accumulator[layer_name].append(attr_aggregated.detach().cpu())
        
        # Average attributions across batches
        for layer_name, attrs in attribution_accumulator.items():
            # Stack only if shapes are compatible
            try:
                avg_attr = torch.stack(attrs).mean(dim=0)
                self._cached_attributions[layer_name] = avg_attr.to(self.device)
            except RuntimeError:
                # If shapes don't match, just use the last one
                self._cached_attributions[layer_name] = attrs[-1].to(self.device)
        
        self._initial_attributions_computed = True
        self._model.train()
        self.logger.info(f"Computed initial attributions for {len(self._cached_attributions)} layers")

    def update_attributions(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Update attribution scores with new batch data using EMA.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
        """
        if self.captum_calculator is None:
            return
        
        try:
            # Compute new attributions
            new_attributions = self.captum_calculator.compute_all_layer_attributions(
                inputs, targets
            )
            
            # Update cache with EMA - aggregate to remove batch dimension first
            for layer_name, new_attr in new_attributions.items():
                # Aggregate along batch dimension to get consistent shape
                if new_attr.dim() >= 2:
                    # Sum over batch dimension
                    new_attr_aggregated = new_attr.abs().sum(dim=0) / new_attr.shape[0]
                else:
                    new_attr_aggregated = new_attr.abs()
                
                if layer_name in self._cached_attributions:
                    old_attr = self._cached_attributions[layer_name]
                    # Ensure shapes match
                    if old_attr.shape == new_attr_aggregated.shape:
                        self._cached_attributions[layer_name] = (
                            self.ema_decay * old_attr + (1 - self.ema_decay) * new_attr_aggregated.detach()
                        )
                    else:
                        # Shape mismatch, just replace
                        self._cached_attributions[layer_name] = new_attr_aggregated.detach()
                else:
                    self._cached_attributions[layer_name] = new_attr_aggregated.detach()
        except Exception as e:
            # Don't let attribution failures break training
            self.logger.debug(f"Attribution update failed: {e}")

    def compute_importance_scores(self, **kwargs):
        """
        Compute importance scores using xAI attributions.
        
        Overrides the base class method to incorporate Captum attributions.
        """
        global_start_idx = 0
        self.global_scores = list()
        
        # Calculate raw importance scores using xAI method
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                calculate_xai_importance_score(
                    self.importance_score_criteria,
                    group,
                    captum_calculator=self.captum_calculator,
                    cached_attributions=self._cached_attributions,
                )

        # Normalize importance scores (same as base class)
        normalization_denoms = dict.fromkeys(
            self.importance_score_criteria.keys(), self.safe_guard
        )
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                for proxy_name in self.importance_score_criteria:
                    if proxy_name not in group["importance_scores"]:
                        continue
                    normalization_denoms[proxy_name] += torch.sum(
                        group["importance_scores"][proxy_name] ** 2, dim=0
                    ).item()
        
        for proxy_name in normalization_denoms:
            normalization_denoms[proxy_name] = (
                np.sqrt(normalization_denoms[proxy_name]) + self.safe_guard
            )

        # Compute overall score
        global_start_idx = 0
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                group["importance_scores"]["overall"] = None
                for proxy_name in self.importance_score_criteria:
                    if proxy_name not in group["importance_scores"]:
                        continue
                    group["importance_scores"][proxy_name].mul_(
                        self.importance_score_criteria[proxy_name]
                        / normalization_denoms[proxy_name]
                    )
                    if group["importance_scores"]["overall"] is None:
                        group["importance_scores"]["overall"] = group[
                            "importance_scores"
                        ][proxy_name].clone()
                    else:
                        group["importance_scores"]["overall"] += group[
                            "importance_scores"
                        ][proxy_name]
                group["global_start_idx"] = global_start_idx
                group["global_idxes"] = np.arange(
                    global_start_idx, global_start_idx + group["num_groups"]
                )
                global_start_idx += group["num_groups"]
                self.global_scores.append(group["importance_scores"]["overall"])

    def identify_redundant_groups(self):
        """Identify redundant groups based on global importance scores."""
        global_importance_scores = torch.cat(self.global_scores, dim=0)
        _, sorted_idx = torch.sort(global_importance_scores, descending=False)
        
        redundant_group_idxes = sorted_idx[:self.target_num_redundant_groups].cpu().numpy()
        
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                global_idxes = group["global_idxes"]
                group_redundant_mask = np.isin(global_idxes, redundant_group_idxes)
                group["active_redundant_idxes"] = np.where(group_redundant_mask)[0].tolist()
                group["important_idxes"] = np.where(~group_redundant_mask)[0].tolist()

    def commit_redundant_idxes(self):
        """Commit active redundant indices to pruned indices."""
        for group in self.param_groups:
            if group["is_prunable"] and not group["is_auxiliary"]:
                group["pruned_idxes"] = list(
                    set(group["pruned_idxes"] + group["active_redundant_idxes"])
                )
                group["active_redundant_idxes"] = []

    def step(self, closure=None, inputs=None, targets=None):
        """
        Core function.
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            inputs: Input tensor for attribution computation (optional)
            targets: Target tensor for attribution computation (optional)
        """
        if closure is not None:
            _ = closure()

        self.num_steps += 1
        self.compute_grad_variant()
        
        # Update attributions periodically during training
        should_update_attr = (
            inputs is not None 
            and targets is not None 
            and self.captum_calculator is not None
            and (self.num_steps - self._last_attribution_step) >= self.compute_attribution_freq
        )
        
        if should_update_attr:
            self.update_attributions(inputs, targets)
            self._last_attribution_step = self.num_steps

        # Determine the bit range projection for weights
        if (
            self.num_steps >= self.start_projection_step
            and self.num_steps <= self.start_pruning_step
            and self.start_projection_step != self.start_pruning_step
        ):
            if (
                self.num_steps - self.start_projection_step - 1
            ) % self.projection_period_duration == 0 and (
                self.num_steps - self.start_projection_step - 1
            ) != 0:
                self.max_bit_wt = self.max_bit_wt - self.bit_reduction
                self.min_bit_wt = self.min_bit_wt

        # Partition groups into important and redundant groups
        if (
            self.num_steps >= self.start_pruning_step
            and self.curr_pruning_period < self.pruning_periods
            and self.pruning_period_duration != 0
        ):
            if (
                self.num_steps - self.start_pruning_step - 1
            ) % self.pruning_period_duration == 0:
                self.logger.info(
                    f"Determining important and redundant groups using xAI scores. Step={self.num_steps}"
                )
                self.commit_redundant_idxes()
                self.compute_importance_scores()
                self.identify_redundant_groups()
                self.curr_pruning_period += 1

        # Update parameters
        if self.pruning_period_duration != 0:
            t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
        
        for group in self.param_groups:
            if not group["is_prunable"] or len(group["active_redundant_idxes"]) == 0:
                if self.num_steps <= self.start_projection_step:
                    self.gradient_descent_step(group)
                elif self.num_steps > self.start_pruning_step + self.pruning_steps:
                    if self.num_steps == self.start_pruning_step + self.pruning_steps + 1:
                        bit_layer = self.get_bitwidth_dict(group)
                        self.bit_layers.update(bit_layer)
                    self.partial_projected_gradient_descent_step_fix(group, self.bit_layers)
                else:
                    self.partial_projected_gradient_descent_step_range_wt(group)
            elif group["is_prunable"] and len(group["active_redundant_idxes"]) > 0:
                # Joint pruning and quantization stage
                for p_name, p, p_transform in zip(
                    group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue
                    if "t_quant_wt" in p_name or "q_m_wt" in p_name:
                        p.data.add_(
                            group["grad_variant"][p_name], alpha=-group["lr_quant"]
                        )

                active_redundant_idxes = group["active_redundant_idxes"]

                # Compute forget rate (gamma) and quant step size (d)
                gamma, d_quant = self.compute_gamma_d(
                    group, active_redundant_idxes, [self.min_bit_wt, self.max_bit_wt]
                )
                self.gamma, self.d_quant = gamma, d_quant

                # Update quant step size d
                for i, (p_name, p_transform) in enumerate(
                    zip(group["p_names"], group["p_transform"])
                ):
                    if "d_quant_wt" in p_name:
                        with torch.no_grad():
                            group["params"][i].copy_(d_quant)

                for p_name, p, p_transform in zip(
                    group["p_names"], group["params"], group["p_transform"]
                ):
                    if p_name not in group["grad_variant"]:
                        continue

                    is_quantize, quantize_weight = self.quantize_weight(group, p_name)
                    if p_transform != TensorTransform.NO_PRUNE:
                        if is_quantize:
                            p.data[active_redundant_idxes] = (
                                p.data[active_redundant_idxes]
                                - gamma * quantize_weight.data[active_redundant_idxes]
                            )
                        else:
                            p.data[active_redundant_idxes] = (
                                p.data[active_redundant_idxes]
                                - gamma * p.data[active_redundant_idxes]
                            )

                    if (
                        "d_quant" not in p_name
                        and "t_quant" not in p_name
                        and "q_m" not in p_name
                    ):
                        p.data.add_(group["grad_variant"][p_name], alpha=-group["lr"])

                    # Tackle auxiliary params
                    for ng_id, offset in group["auxiliary_ngs"]:
                        active_redundant_aux_idxes = [
                            i + offset for i in active_redundant_idxes
                        ]
                        for aux_p in self.auxiliary_param_groups[ng_id]["params"]:
                            if aux_p.grad is None:
                                continue
                            aux_p.data[active_redundant_aux_idxes, ...] *= (
                                self.pruning_period_duration - t - 1.0
                            ) / (self.pruning_period_duration - t)

            self.fix_pruned_groups_as_zeros(group)

        if self.pruning_period_duration != 0:
            if self.num_steps >= self.start_pruning_step and t == self.pruning_period_duration - 1:
                self.commit_redundant_idxes()

    # =====================================================================
    # Helper methods (copied from GETA with minor modifications)
    # =====================================================================
    
    def gradient_descent_step(self, param_group):
        """Standard gradient descent step."""
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            if "d_quant" in p_name or "t_quant" in p_name or "q_m" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

    def quantize_weight(self, param_group, target_name):
        """Quantize weight for a given parameter."""
        is_quantize = False
        t_quant = None
        
        for p_name in param_group["p_names"]:
            if "d_quant_wt" in p_name:
                layer_name = ".".join(p_name.split(".")[:-1])
                if layer_name in target_name and "weight" in target_name:
                    is_quantize = True
                    quantize_layer_name = layer_name
                    break

        if not is_quantize:
            return is_quantize, None
        else:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if quantize_layer_name in p_name and "d_quant_wt" in p_name:
                    d_quant = p.data
                if quantize_layer_name in p_name and "t_quant_wt" in p_name:
                    t_quant = p.data
                if quantize_layer_name in p_name and "q_m_wt" in p_name:
                    q_m = p.data
                if quantize_layer_name in p_name and "weight" in p_name:
                    weight = p.data

            with torch.no_grad():
                quantized_weight = self._quantize_helper(
                    weight, d_quant, q_m, t_quant=t_quant
                )

            return is_quantize, quantized_weight

    def compute_gamma_d(self, param_group, active_redundant_idxes, bit_range):
        """Compute forget rate (gamma) and quantization step size (d)."""
        t_quant = None
        qm_list = []
        layer_name_list = []
        prune_param_clip_list = []
        prune_param_grad_list = []
        prune_param_res_list = []
        prune_param_clip_redundant_list = []
        prune_param_res_redundant_list = []
        prune_param_grad_redundant_list = []

        # Layers with quantization mapping
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "d_quant_wt" in p_name:
                        d_quant = p.data
                    if "t_quant_wt" in p_name:
                        t_quant = p.data
                    if "q_m_wt" in p_name:
                        q_m = p.data
                        qm_list.append(q_m.item())
                    if "weight" in p_name:
                        weight = p.data
            
            for p_name, p, p_transform in zip(
                param_group["p_names"],
                param_group["params"],
                param_group["p_transform"],
            ):
                if layer_name in p_name and "weight" in p_name:
                    clipped_weight = self._clip_helper(weight, q_m, t_quant=t_quant)
                    prune_param_clip_list.append(clipped_weight.data)
                    residual_weight = self._residual_helper(
                        weight, d_quant, q_m, t_quant=t_quant
                    )
                    prune_param_res_list.append(residual_weight)
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])
                elif layer_name in p_name and p_transform != 1:
                    prune_param_clip_list.append(p.data)
                    prune_param_res_list.append(
                        torch.tensor([0.0]).to(p.device).expand_as(p.data)
                    )
                    prune_param_grad_list.append(param_group["grad_variant"][p_name])

        # Layers without quantization mapping
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if not any(layer_name in p_name for layer_name in layer_name_list):
                prune_param_clip_list.append(p.data)
                prune_param_res_list.append(
                    torch.tensor([0.0]).to(p.device).expand_as(p.data)
                )
                prune_param_grad_list.append(param_group["grad_variant"][p_name])

        # Access values at redundant indices
        for i in range(len(prune_param_clip_list)):
            prune_param_clip_redundant_list.append(
                prune_param_clip_list[i][active_redundant_idxes]
            )
            prune_param_res_redundant_list.append(
                prune_param_res_list[i][active_redundant_idxes]
            )
            prune_param_grad_redundant_list.append(
                prune_param_grad_list[i][active_redundant_idxes]
            )

        flatten_clip = torch.cat(
            [tensor.flatten() for tensor in prune_param_clip_redundant_list]
        )
        flatten_grad = torch.cat(
            [tensor.flatten() for tensor in prune_param_grad_redundant_list]
        )
        flatten_res = torch.cat(
            [tensor.flatten() for tensor in prune_param_res_redundant_list]
        )
        flatten_clip_norm = torch.norm(flatten_clip, p=2)
        flatten_grad_norm = torch.norm(flatten_grad, p=2)
        flatten_res_norm = torch.norm(flatten_res, p=2)

        eps = 1e-8
        cosine_similarity_clip = torch.div(
            torch.dot(flatten_clip, flatten_grad),
            torch.max(flatten_clip_norm, torch.tensor(eps).to(flatten_clip.device))
            * flatten_grad_norm,
        )
        cosine_similarity_res = torch.div(
            torch.dot(flatten_res, flatten_grad),
            torch.max(flatten_res_norm, torch.tensor(eps).to(flatten_res.device))
            * flatten_grad_norm,
        )

        eta = 0.999
        zeta = 0.9
        
        if torch.mean(flatten_clip).item() < 1e-8:
            forget_rate = 0.0
        else:
            if torch.isinf(cosine_similarity_clip) or torch.isnan(cosine_similarity_clip):
                forget_rate = 0.0
            elif cosine_similarity_clip >= 0.0 and cosine_similarity_clip <= 1.0:
                t = (self.num_steps - self.start_pruning_step) % self.pruning_period_duration
                forget_rate = 1.0 - (self.pruning_period_duration - t - 1.0) / (
                    self.pruning_period_duration - t
                )
            elif cosine_similarity_clip >= -1.0 and cosine_similarity_clip < 0.0:
                forget_rate = (
                    -(1 - eta)
                    * param_group["lr"]
                    * flatten_grad_norm
                    / (cosine_similarity_clip * flatten_clip_norm)
                )
            else:
                forget_rate = 0.0

        # Determine d_quant range
        bit_width_lower = bit_range[0]
        bit_width_upper = bit_range[1]
        d_quant_upper = self._d_quant_helper(
            bit_width_lower, max(np.abs(qm_list)) if qm_list else 1.0, t_quant
        )
        d_quant_lower = self._d_quant_helper(
            bit_width_upper, max(np.abs(qm_list)) if qm_list else 1.0, t_quant
        )

        if cosine_similarity_res >= 0.0 or forget_rate == 0.0:
            d_quant = d_quant_upper
        else:
            d_quant = (
                -zeta
                * eta
                * param_group["lr"]
                * flatten_grad_norm
                / (forget_rate * cosine_similarity_res * flatten_res_norm + eps)
            )
            # Safeguard: if d_quant is negative or non-finite, use d_quant_upper
            # Handle both tensor and scalar cases
            if isinstance(d_quant, torch.Tensor):
                d_quant_val = d_quant.item()
            else:
                d_quant_val = d_quant
            
            if d_quant_val <= 0 or not np.isfinite(d_quant_val):
                d_quant = d_quant_upper
            else:
                # Add max iterations to prevent infinite loop
                # Use scalar value for comparison
                max_iters = 100
                iters = 0
                d_quant = d_quant_val  # Convert to scalar for loop
                while d_quant < d_quant_lower and iters < max_iters:
                    forget_rate = forget_rate * 0.8
                    d_quant = d_quant / 0.8
                    iters += 1
                if iters >= max_iters:
                    d_quant = d_quant_lower  # Use lower bound if loop didn't converge
            d_quant = min(d_quant_upper, d_quant)

        return forget_rate, d_quant

    def get_bitwidth_dict(self, param_group):
        """Get bit width dictionary for layers."""
        layer_name_list = []
        bit_dict = {}
        
        for p_name in param_group["p_names"]:
            if "d_quant" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            d_quant_wt = None
            q_m_wt = None
            t_quant_wt = None
            d_quant_act = None
            q_m_act = None
            t_quant_act = None
            
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data
                    if "d_quant_wt" in p_name:
                        d_quant_wt = p.data
                    if "t_quant_act" in p_name:
                        t_quant_act = p.data
                    if "q_m_act" in p_name:
                        q_m_act = p.data
                    if "d_quant_act" in p_name:
                        d_quant_act = p.data

            bit_dict[layer_name] = {}
            bit_width_wt = self._bit_width_helper(
                d_quant=d_quant_wt, q_m=q_m_wt, t_quant=t_quant_wt
            )
            bit_width_act = self._bit_width_helper(
                d_quant=d_quant_act, q_m=q_m_act, t_quant=t_quant_act
            )

            if bit_width_wt is not None:
                # Safety check for NaN (should not happen after fix, but just in case)
                if np.isnan(bit_width_wt):
                    bit_width_wt = 8.0
                bit_dict[layer_name]["weight"] = round(bit_width_wt)
            if bit_width_act is not None:
                # Safety check for NaN (should not happen after fix, but just in case)
                if np.isnan(bit_width_act):
                    bit_width_act = 8.0
                bit_dict[layer_name]["activation"] = round(bit_width_act)

        return bit_dict

    def partial_projected_gradient_descent_step_range_wt(self, param_group):
        """Apply projected gradient descent for weight quantization parameters."""
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

        layer_name_list = []
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            t_quant_wt = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data

            d_quant_min = self._d_quant_helper(self.max_bit_wt, q_m_wt, t_quant_wt)
            d_quant_max = self._d_quant_helper(self.min_bit_wt, q_m_wt, t_quant_wt)

            if isinstance(d_quant_min, torch.Tensor):
                d_quant_min = float(d_quant_min.item())
            if isinstance(d_quant_max, torch.Tensor):
                d_quant_max = float(d_quant_max.item())

            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name and "d_quant_wt" in p_name:
                    p.data.clamp_(min=d_quant_min, max=d_quant_max)

    def partial_projected_gradient_descent_step_fix(self, param_group, bit_dict):
        """Apply projected gradient descent with fixed bit width."""
        for p_name, p in zip(param_group["p_names"], param_group["params"]):
            if p_name not in param_group["grad_variant"]:
                continue
            if (
                param_group["weight_decay"] is not None
                and param_group["variant"] == "adamw"
            ):
                if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                    p.data.add_(
                        param_group["weight_decay"] * p.data,
                        alpha=-param_group["lr_quant"],
                    )
                else:
                    p.data.add_(
                        param_group["weight_decay"] * p.data, alpha=-param_group["lr"]
                    )

            if "d_quant_wt" in p_name or "t_quant_wt" in p_name or "q_m_wt" in p_name:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr_quant"]
                )
            else:
                p.data.add_(
                    param_group["grad_variant"][p_name], alpha=-param_group["lr"]
                )

        layer_name_list = []
        for p_name in param_group["p_names"]:
            if "d_quant_wt" not in p_name:
                continue
            layer_name = ".".join(p_name.split(".")[:-1])
            if layer_name not in layer_name_list:
                layer_name_list.append(layer_name)

        for layer_name in layer_name_list:
            if layer_name not in bit_dict:
                continue
            t_quant_wt = None
            for p_name, p in zip(param_group["p_names"], param_group["params"]):
                if layer_name in p_name:
                    if "t_quant_wt" in p_name:
                        t_quant_wt = p.data
                    if "q_m_wt" in p_name:
                        q_m_wt = p.data

            if "weight" in bit_dict[layer_name]:
                target_bit = bit_dict[layer_name]["weight"]
                d_quant_target = self._d_quant_helper(target_bit, q_m_wt, t_quant_wt)

                if isinstance(d_quant_target, torch.Tensor):
                    d_quant_target = float(d_quant_target.item())

                for p_name, p in zip(param_group["p_names"], param_group["params"]):
                    if layer_name in p_name and "d_quant_wt" in p_name:
                        p.data.fill_(d_quant_target)

    # =====================================================================
    # Quantization helper methods
    # =====================================================================
    
    def _bit_width_helper(self, d_quant, q_m, t_quant=None):
        """Convert d_quant to bit width.
        
        Returns a valid bit width (1-16) or None if computation fails.
        Handles NaN and invalid values gracefully.
        """
        if d_quant is None or q_m is None:
            return None
        
        try:
            if t_quant is not None:
                q_max = torch.exp(t_quant * torch.log(q_m))
            else:
                q_max = q_m
            
            if isinstance(d_quant, torch.Tensor):
                d_quant = d_quant.item()
            if isinstance(q_max, torch.Tensor):
                q_max = q_max.item()
            
            # Check for NaN or invalid values
            if np.isnan(d_quant) or np.isnan(q_max):
                return 8.0  # Default to 8-bit if values are NaN
            
            if d_quant <= 0:
                return 16.0
            
            if q_max <= 0:
                return 8.0  # Default to 8-bit if q_max is invalid
            
            ratio = q_max / d_quant + 1
            if ratio <= 0 or np.isnan(ratio) or np.isinf(ratio):
                return 8.0  # Default to 8-bit if ratio is invalid
            
            bit_width = np.log2(ratio) + 1
            
            # Check for NaN or infinity in result
            if np.isnan(bit_width) or np.isinf(bit_width):
                return 8.0  # Default to 8-bit
            
            # Clamp bit width to valid range
            bit_width = max(1.0, min(16.0, float(bit_width)))
            return bit_width
            
        except Exception as e:
            # If any error occurs, return default 8-bit
            return 8.0

    def _d_quant_helper(self, bit_width, q_m, t_quant=None):
        """Convert bit width to d_quant."""
        if isinstance(q_m, torch.Tensor):
            q_m_val = q_m.item()
        else:
            q_m_val = q_m
        
        if t_quant is not None:
            if isinstance(t_quant, torch.Tensor):
                t_quant_val = t_quant.item()
            else:
                t_quant_val = t_quant
            q_max = np.exp(t_quant_val * np.log(q_m_val))
        else:
            q_max = q_m_val
        
        d_quant = q_max / (2 ** (bit_width - 1) - 1)
        return d_quant

    def _quantize_helper(self, weight, d_quant, q_m, t_quant=None):
        """Quantize weight."""
        if t_quant is not None:
            q_max = torch.exp(t_quant * torch.log(q_m))
        else:
            q_max = q_m
        
        clipped = torch.clamp(weight, -q_max, q_max)
        quantized = d_quant * torch.round(clipped / d_quant)
        return quantized

    def _clip_helper(self, weight, q_m, t_quant=None):
        """Clip weight to quantization range."""
        if t_quant is not None:
            q_max = torch.exp(t_quant * torch.log(q_m))
        else:
            q_max = q_m
        
        return torch.clamp(weight, -q_max, q_max)

    def _residual_helper(self, weight, d_quant, q_m, t_quant=None):
        """Compute quantization residual."""
        clipped = self._clip_helper(weight, q_m, t_quant)
        quantized = self._quantize_helper(weight, d_quant, q_m, t_quant)
        return clipped - quantized

    def compute_metrics(self):
        """Compute optimizer metrics."""
        self.opt_metrics.norm_params = 0.0
        self.opt_metrics.norm_important_groups = 0.0
        self.opt_metrics.norm_redundant_groups = 0.0
        self.opt_metrics.num_zero_groups = 0
        self.opt_metrics.num_important_groups = 0
        self.opt_metrics.num_redundant_groups = 0

        for group in self.param_groups:
            if not (group["is_prunable"] and not group["is_auxiliary"]):
                continue
            norm_group = None
            import_idxes = group["important_idxes"]
            redund_idxes = group["active_redundant_idxes"] + group["pruned_idxes"]

            for param, p_transform in zip(group["params"], group["p_transform"]):
                if p_transform == TensorTransform.NO_PRUNE:
                    continue
                param_transform = tensor_transformation_param_group(param.data, p_transform, group)
                if norm_group is None:
                    norm_group = torch.norm(param_transform, dim=1) ** 2
                else:
                    norm_group += torch.norm(param_transform, dim=1) ** 2
            
            if norm_group is not None:
                norm_group = torch.sqrt(norm_group)
                self.opt_metrics.num_zero_groups += torch.sum(norm_group == 0).item()
                self.opt_metrics.norm_params += torch.sum(norm_group).item()
                self.opt_metrics.norm_important_groups += torch.sum(
                    norm_group[import_idxes]
                ).item()
                self.opt_metrics.norm_redundant_groups += torch.sum(
                    norm_group[redund_idxes]
                ).item()
            self.opt_metrics.num_important_groups += len(import_idxes)
            self.opt_metrics.num_redundant_groups += len(redund_idxes)

        self.opt_metrics.group_sparsity = self.opt_metrics.num_zero_groups / float(
            self.total_num_groups + self.safe_guard
        )

        return self.opt_metrics
