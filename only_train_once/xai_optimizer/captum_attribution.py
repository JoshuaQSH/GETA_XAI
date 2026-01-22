"""
Captum Attribution Calculator for GETA

This module wraps PyTorch Captum attribution methods for computing
layer-wise importance scores for structured pruning.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Union

import torch
import torch.nn as nn

# Include the attribution methods -> maps to WISDOM
try:
    from captum.attr import (
        IntegratedGradients,
        LayerIntegratedGradients,
        LayerGradientXActivation,
        LayerConductance,
        LayerActivation,
        Saliency,
        InputXGradient,
        GuidedBackprop,
        DeepLift,
        LayerDeepLift,
        GradientShap,
        LayerGradientShap,
        LRP,
        LayerLRP,
        Deconvolution,
        GuidedGradCam,
        LayerGradCam,
    )
    CAPTUM_AVAILABLE = True
    LRP_AVAILABLE = True
except ImportError as e:
    CAPTUM_AVAILABLE = False
    LRP_AVAILABLE = False
    print(f"Warning: Captum not installed or partial import error: {e}")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Methods that require inplace=False on ReLU layers
LRP_METHODS = ['lrp', 'layer_lrp', 'guided_backprop', 'deconvolution']

# Methods that modify model hooks and need special handling with quantization
# These methods will use a deep copy of the model to avoid interfering with training
HOOK_MODIFYING_METHODS = ['guided_backprop']

# Methods that are truly incompatible with quantization (none currently - we handle guided_backprop with deep copy)
QUANTIZATION_INCOMPATIBLE_METHODS = []


class AttributionMethod(Enum):
    """
    Supported Captum attribution methods for structured pruning.
    
    Categories:
    -----------
    1. GRADIENT-BASED (fast, single backward pass):
       - SALIENCY: Gradient magnitude w.r.t. input
       - INPUT_X_GRADIENT: Input * gradient (simple sensitivity)
       - GUIDED_BACKPROP: Modified backprop that only propagates positive gradients
    
    2. PATH-BASED (slower, require baseline and integration):
       - INTEGRATED_GRADIENTS: Path integral from baseline to input
       - DEEP_LIFT: Difference from reference propagation
       - GRADIENT_SHAP: SHAP values via gradient sampling
    
    3. LAYER-SPECIFIC (compute attribution for specific layers):
       - LAYER_CONDUCTANCE: How much a layer affects output (like integrated gradients for layers)
       - LAYER_GRADIENT_X_ACTIVATION: Gradient * activation at a layer
       - LAYER_INTEGRATED_GRADIENTS: Integrated gradients at layer level
       - LAYER_ACTIVATION: Raw activations (not attribution, just feature importance)
       - LAYER_LRP: Layer-wise Relevance Propagation for specific layers
       - LAYER_GRADCAM: Grad-CAM for layer visualization
    
    4. DECOMPOSITION-BASED:
       - LRP: Layer-wise Relevance Propagation (propagates relevance backward)
       - DECONVOLUTION: Deconvolution-based visualization
    
    NOT INCLUDED:
    - LayerActivation: Returns raw activations, not attributions. Activations alone don't 
      measure importance - they need to be combined with gradients or other signals.
    """
    # Gradient-based methods (fast)
    SALIENCY = "saliency"
    INPUT_X_GRADIENT = "input_x_gradient"
    GUIDED_BACKPROP = "guided_backprop"
    
    # Path-based methods (slower, more accurate)
    INTEGRATED_GRADIENTS = "integrated_gradients"
    DEEP_LIFT = "deep_lift"
    GRADIENT_SHAP = "gradient_shap"
    
    # Layer-specific methods
    LAYER_CONDUCTANCE = "layer_conductance"
    LAYER_GRADIENT_X_ACTIVATION = "layer_gradient_x_activation"
    LAYER_INTEGRATED_GRADIENTS = "layer_integrated_gradients"
    LAYER_ACTIVATION = "layer_activation"  # Raw activations (not true attribution)
    LAYER_LRP = "layer_lrp"
    LAYER_GRADCAM = "layer_gradcam"
    
    # Decomposition-based methods
    LRP = "lrp"
    DECONVOLUTION = "deconvolution"


class CaptumAttributionCalculator:
    """
    Calculator for structured pruning importance scores using Captum attributions.
    
    This class wraps Captum attribution methods to compute per-group importance
    scores for structured pruning in GETA.
    
    Attributes:
        model: The neural network model
        attribution_method: Captum method to use
        baseline_type: Type of baseline for attribution
        n_steps: Number of steps for integrated gradients methods
        device: Device to run computations on
    """
    
    def __init__(
        self,
        model: nn.Module,
        attribution_method: str = "saliency",  # Changed default to saliency for lower memory
        baseline_type: str = "zero",
        n_steps: int = 10,  # Reduced from 20
        device: str = "cuda",
    ):
        """
        Initialize the Captum Attribution Calculator.
        
        Args:
            model: The neural network model
            attribution_method: Captum method to use (see AttributionMethod)
            baseline_type: Type of baseline for attribution ("zero", "random", "mean")
            n_steps: Number of steps for integrated gradients methods
            device: Device to run computations on
        """
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum is required. Install with: pip install captum")
        
        self.model = model
        self.attribution_method = attribution_method
        self.baseline_type = baseline_type
        self.n_steps = n_steps
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for layer attributors
        self._layer_attributors: Dict[str, object] = {}
        
        # Build layer name to module mapping
        self._layer_to_module: Dict[str, nn.Module] = {}
        self._param_to_layer: Dict[str, str] = {}
        self._build_layer_mapping()
        
        # Cache for attribution scores
        self._attribution_cache: Dict[str, torch.Tensor] = {}
        
    def _build_layer_mapping(self):
        """Build mapping from layer names to modules and parameters."""
        for name, module in self.model.named_modules():
            self._layer_to_module[name] = module
            
        for name, param in self.model.named_parameters():
            # Extract layer name from parameter name (e.g., "layer1.conv.weight" -> "layer1.conv")
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                layer_name = parts[0]
                self._param_to_layer[name] = layer_name
    
    def _get_layer_module(self, layer_name: str) -> Optional[nn.Module]:
        """Get the module for a given layer name."""
        return self._layer_to_module.get(layer_name)
    
    def _get_layer_from_param(self, param_name: str) -> Optional[str]:
        """Get the layer name from a parameter name."""
        return self._param_to_layer.get(param_name)
    
    def _create_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Create baseline input for attribution methods."""
        if self.baseline_type == "zero":
            return torch.zeros_like(input_tensor)
        elif self.baseline_type == "random":
            return torch.randn_like(input_tensor) * 0.01
        elif self.baseline_type == "mean":
            return torch.ones_like(input_tensor) * input_tensor.mean()
        else:
            return torch.zeros_like(input_tensor)
    
    def _get_attributor(self, layer_name: str, method: str):
        """Get or create an attributor for a specific layer."""
        cache_key = f"{layer_name}_{method}"
        
        if cache_key in self._layer_attributors:
            return self._layer_attributors[cache_key]
        
        layer_module = self._get_layer_module(layer_name)
        if layer_module is None:
            self.logger.warning(f"Layer module not found for {layer_name}")
            return None
        
        # Create forward function for the model
        def forward_func(x):
            return self.model(x)
        
        try:
            # ============ LAYER-SPECIFIC METHODS ============
            if method == AttributionMethod.LAYER_CONDUCTANCE.value:
                attributor = LayerConductance(forward_func, layer_module)
            elif method == AttributionMethod.LAYER_GRADIENT_X_ACTIVATION.value:
                attributor = LayerGradientXActivation(forward_func, layer_module)
            elif method == AttributionMethod.LAYER_INTEGRATED_GRADIENTS.value:
                attributor = LayerIntegratedGradients(forward_func, layer_module)
            elif method == AttributionMethod.DEEP_LIFT.value:
                # DeepLift needs the actual model (nn.Module), not just forward function
                attributor = LayerDeepLift(self.model, layer_module)
            elif method == AttributionMethod.LAYER_ACTIVATION.value:
                # NOTE: LayerActivation returns raw activations, not attributions
                # Use this for baseline comparison - activations alone don't measure importance
                attributor = LayerActivation(forward_func, layer_module)
            elif method == AttributionMethod.LAYER_LRP.value:
                # Layer-wise Relevance Propagation for specific layers
                # NOTE: LRP requires model without in-place operations (e.g., ReLU(inplace=False))
                # LRP needs the actual model (nn.Module)
                attributor = LayerLRP(self.model, layer_module)
            elif method == AttributionMethod.LAYER_GRADCAM.value:
                # Grad-CAM: Works best with conv layers
                attributor = LayerGradCam(forward_func, layer_module)
            
            # ============ INPUT-LEVEL METHODS ============
            elif method == AttributionMethod.SALIENCY.value:
                attributor = Saliency(forward_func)
            elif method == AttributionMethod.INPUT_X_GRADIENT.value:
                attributor = InputXGradient(forward_func)
            elif method == AttributionMethod.INTEGRATED_GRADIENTS.value:
                attributor = IntegratedGradients(forward_func)
            elif method == AttributionMethod.GUIDED_BACKPROP.value:
                # GuidedBackprop modifies ReLU backward hooks, which can conflict with
                # quantization layers. We use a deep copy of the model to isolate the
                # attribution computation from the training model.
                import copy
                model_copy = copy.deepcopy(self.model)
                model_copy.eval()
                attributor = GuidedBackprop(model_copy)
                # Store the copy so it can be garbage collected properly
                self._guided_backprop_model = model_copy
            elif method == AttributionMethod.GRADIENT_SHAP.value:
                # NOTE: GradientShap is slow - requires many baseline samples
                attributor = GradientShap(forward_func)
            
            # ============ DECOMPOSITION-BASED METHODS ============
            elif method == AttributionMethod.LRP.value:
                # Layer-wise Relevance Propagation (full model)
                # NOTE: Requires model without in-place operations (ReLU inplace=False)
                # LRP needs the actual model (nn.Module)
                attributor = LRP(self.model)
            elif method == AttributionMethod.DECONVOLUTION.value:
                # Deconvolution needs the actual model (nn.Module)
                attributor = Deconvolution(self.model)
            
            else:
                # Default to layer conductance for unknown methods
                self.logger.warning(f"Unknown method {method}, defaulting to layer_conductance")
                attributor = LayerConductance(forward_func, layer_module)
            
            self._layer_attributors[cache_key] = attributor
            return attributor
            
        except Exception as e:
            self.logger.warning(f"Failed to create attributor for {layer_name}: {e}")
            return None
    
    def compute_layer_attribution(
        self,
        input_tensor: torch.Tensor,
        target: Union[torch.Tensor, int],
        layer_name: str,
    ) -> Optional[torch.Tensor]:
        """
        Compute attribution scores for a specific layer.
        
        Args:
            input_tensor: Model input (batch of samples)
            target: Target labels or class index
            layer_name: Name of the layer to compute attributions for
            
        Returns:
            Attribution tensor or None if computation fails
        """
        attributor = self._get_attributor(layer_name, self.attribution_method)
        if attributor is None:
            return None
        
        baseline = self._create_baseline(input_tensor)
        
        # Convert target to appropriate format
        if isinstance(target, torch.Tensor):
            if target.dim() > 0:
                target = target[0].item() if target.numel() == 1 else target
        
        try:
            self.model.eval()  # Ensure model is in eval mode for attribution
            
            # ============ METHODS REQUIRING BASELINE + STEPS ============
            if self.attribution_method in [
                AttributionMethod.LAYER_CONDUCTANCE.value,
                AttributionMethod.LAYER_INTEGRATED_GRADIENTS.value,
            ]:
                attributions = attributor.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target,
                    n_steps=self.n_steps,
                )
            
            # ============ METHODS REQUIRING BASELINE (NO STEPS) ============
            elif self.attribution_method in [
                AttributionMethod.DEEP_LIFT.value,
                AttributionMethod.INTEGRATED_GRADIENTS.value,
            ]:
                attributions = attributor.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target,
                )
            
            # ============ GRADIENT-BASED METHODS (NO BASELINE) ============
            elif self.attribution_method in [
                AttributionMethod.SALIENCY.value,
                AttributionMethod.INPUT_X_GRADIENT.value,
                AttributionMethod.GUIDED_BACKPROP.value,
                AttributionMethod.DECONVOLUTION.value,
            ]:
                attributions = attributor.attribute(input_tensor, target=target)
            
            # ============ LAYER-SPECIFIC METHODS ============
            elif self.attribution_method == AttributionMethod.LAYER_GRADIENT_X_ACTIVATION.value:
                attributions = attributor.attribute(
                    input_tensor,
                    target=target,
                )
            elif self.attribution_method == AttributionMethod.LAYER_ACTIVATION.value:
                # LayerActivation doesn't need target - just returns activations
                attributions = attributor.attribute(input_tensor)
            elif self.attribution_method == AttributionMethod.LAYER_LRP.value:
                # LayerLRP needs target for relevance propagation
                attributions = attributor.attribute(input_tensor, target=target)
            elif self.attribution_method == AttributionMethod.LAYER_GRADCAM.value:
                # Grad-CAM needs target
                attributions = attributor.attribute(input_tensor, target=target)
            
            # ============ LRP (FULL MODEL) ============
            elif self.attribution_method == AttributionMethod.LRP.value:
                attributions = attributor.attribute(input_tensor, target=target)
            
            # ============ GRADIENT SHAP (SLOW - MULTIPLE BASELINES) ============
            elif self.attribution_method == AttributionMethod.GRADIENT_SHAP.value:
                # GradientShap requires multiple baselines for sampling
                # Create a small batch of baselines
                baselines = torch.cat([
                    self._create_baseline(input_tensor) for _ in range(5)
                ], dim=0)
                attributions = attributor.attribute(
                    input_tensor,
                    baselines=baselines,
                    target=target,
                    n_samples=20,  # Number of samples for SHAP
                )
            
            else:
                # Default: methods without baseline
                attributions = attributor.attribute(
                    input_tensor,
                    target=target,
                )
            
            self.model.train()  # Return to training mode
            return attributions
            
        except Exception as e:
            self.logger.warning(f"Attribution failed for {layer_name}: {e}")
            self.model.train()
            return None
    
    def compute_all_layer_attributions(
        self,
        input_tensor: torch.Tensor,
        target: Union[torch.Tensor, int],
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute attribution scores for all (or specified) layers.
        
        Args:
            input_tensor: Model input (batch of samples)
            target: Target labels
            layer_names: List of layer names to compute. If None, compute for all layers.
            
        Returns:
            Dictionary mapping layer names to attribution tensors
        """
        if layer_names is None:
            # Get all layers with parameters
            layer_names = list(set(self._param_to_layer.values()))
        
        attributions = {}
        for layer_name in layer_names:
            attr = self.compute_layer_attribution(input_tensor, target, layer_name)
            if attr is not None:
                attributions[layer_name] = attr
        
        return attributions
    
    def aggregate_attribution_to_groups(
        self,
        attribution: torch.Tensor,
        num_groups: int,
        aggregation: str = "sum_abs",
    ) -> torch.Tensor:
        """
        Aggregate layer attribution to per-group importance scores.
        
        For structured pruning, we need to aggregate attributions along the
        channel/group dimension.
        
        Args:
            attribution: Attribution tensor from Captum
            num_groups: Number of pruning groups
            aggregation: Aggregation method ("sum_abs", "mean_abs", "max_abs")
            
        Returns:
            Per-group importance scores tensor of shape [num_groups]
        """
        # Attribution shape depends on the layer type
        # For Conv2d: [batch, channels, height, width]
        # For Linear: [batch, features]
        
        if attribution.dim() == 4:
            # Convolutional layer: aggregate over spatial dimensions and batch
            if aggregation == "sum_abs":
                channel_attr = torch.abs(attribution).sum(dim=(0, 2, 3))  # [channels]
            elif aggregation == "mean_abs":
                channel_attr = torch.abs(attribution).mean(dim=(0, 2, 3))
            else:  # max_abs
                channel_attr = torch.abs(attribution).amax(dim=(0, 2, 3))
        elif attribution.dim() == 2:
            # Linear layer: aggregate over batch
            if aggregation == "sum_abs":
                channel_attr = torch.abs(attribution).sum(dim=0)  # [features]
            elif aggregation == "mean_abs":
                channel_attr = torch.abs(attribution).mean(dim=0)
            else:
                channel_attr = torch.abs(attribution).amax(dim=0)
        elif attribution.dim() == 3:
            # Could be sequence data or other format
            if aggregation == "sum_abs":
                channel_attr = torch.abs(attribution).sum(dim=(0, 2))
            elif aggregation == "mean_abs":
                channel_attr = torch.abs(attribution).mean(dim=(0, 2))
            else:
                channel_attr = torch.abs(attribution).amax(dim=(0, 2))
        else:
            # Flatten and handle
            channel_attr = torch.abs(attribution).flatten()
        
        # Reshape to match num_groups if needed
        if channel_attr.numel() != num_groups:
            if channel_attr.numel() > num_groups:
                # Reshape and sum within groups
                group_size = channel_attr.numel() // num_groups
                if group_size * num_groups == channel_attr.numel():
                    channel_attr = channel_attr[:num_groups * group_size]
                    channel_attr = channel_attr.view(num_groups, group_size).sum(dim=1)
                else:
                    # Truncate or interpolate
                    channel_attr = channel_attr[:num_groups]
            else:
                # Pad with zeros
                padded = torch.zeros(num_groups, device=channel_attr.device)
                padded[:channel_attr.numel()] = channel_attr
                channel_attr = padded
        
        return channel_attr
    
    def update_attribution_cache(
        self,
        layer_name: str,
        new_attribution: torch.Tensor,
        ema_decay: float = 0.9,
    ):
        """
        Update the attribution cache with EMA blending.
        
        Args:
            layer_name: Name of the layer
            new_attribution: New attribution scores
            ema_decay: Exponential moving average decay factor
        """
        if layer_name in self._attribution_cache:
            old_attr = self._attribution_cache[layer_name]
            self._attribution_cache[layer_name] = (
                ema_decay * old_attr + (1 - ema_decay) * new_attribution.detach()
            )
        else:
            self._attribution_cache[layer_name] = new_attribution.detach().clone()
    
    def get_cached_attribution(self, layer_name: str) -> Optional[torch.Tensor]:
        """Get cached attribution for a layer."""
        return self._attribution_cache.get(layer_name)
    
    def clear_cache(self):
        """Clear the attribution cache."""
        self._attribution_cache.clear()
        self._layer_attributors.clear()
