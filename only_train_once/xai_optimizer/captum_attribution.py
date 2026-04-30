"""
Captum attribution helpers for structured pruning in XAI-GETA.

The structured pruning signal must align with prunable groups, so all supported
methods are resolved to layer-aware attributions rather than raw input
attributions.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

try:
    from captum.attr import (
        LayerConductance,
        LayerDeepLift,
        LayerGradientShap,
        LayerGradientXActivation,
        LayerIntegratedGradients,
        LayerLRP,
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


LRP_METHODS = ["lrp", "layer_lrp"]
QUANTIZATION_INCOMPATIBLE_METHODS = ["guided_backprop", "deconvolution"]

SUPPORTED_STRUCTURED_ATTRIBUTION_METHODS = [
    "saliency",
    "input_x_gradient",
    "layer_conductance",
    "layer_gradient_x_activation",
    "layer_integrated_gradients",
    "deep_lift",
    "integrated_gradients",
    "gradient_shap",
    "lrp",
    "layer_lrp",
]

UNSUPPORTED_STRUCTURED_ATTRIBUTION_METHODS = [
    "guided_backprop",
    "deconvolution",
    "layer_activation",
    "layer_gradcam",
]

STRUCTURED_METHOD_ALIASES = {
    "saliency": "layer_saliency",
    "input_x_gradient": "layer_gradient_x_activation",
    "integrated_gradients": "layer_integrated_gradients",
    "gradient_shap": "layer_gradient_shap",
    "lrp": "layer_lrp",
}


class AttributionMethod(Enum):
    SALIENCY = "saliency"
    INPUT_X_GRADIENT = "input_x_gradient"
    GUIDED_BACKPROP = "guided_backprop"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    DEEP_LIFT = "deep_lift"
    GRADIENT_SHAP = "gradient_shap"
    LAYER_CONDUCTANCE = "layer_conductance"
    LAYER_GRADIENT_X_ACTIVATION = "layer_gradient_x_activation"
    LAYER_INTEGRATED_GRADIENTS = "layer_integrated_gradients"
    LAYER_ACTIVATION = "layer_activation"
    LAYER_LRP = "layer_lrp"
    LAYER_GRADCAM = "layer_gradcam"
    LRP = "lrp"
    DECONVOLUTION = "deconvolution"


class CaptumAttributionCalculator:
    """
    Compute layer-aware attributions for structured pruning groups.

    User-facing method names such as ``saliency`` and ``integrated_gradients``
    are mapped to layer-aware variants so the returned tensors align with
    prunable channels/features instead of the raw model input.
    """

    def __init__(
        self,
        model: nn.Module,
        attribution_method: str = "saliency",
        baseline_type: str = "zero",
        n_steps: int = 10,
        device: str = "cuda",
    ):
        if not CAPTUM_AVAILABLE:
            raise RuntimeError("Captum is required. Install with: pip install captum")

        if attribution_method in UNSUPPORTED_STRUCTURED_ATTRIBUTION_METHODS:
            raise ValueError(
                f"{attribution_method} is not supported for structured pruning. "
                f"Use one of: {SUPPORTED_STRUCTURED_ATTRIBUTION_METHODS}"
            )
        if attribution_method not in SUPPORTED_STRUCTURED_ATTRIBUTION_METHODS:
            raise ValueError(
                f"Unknown attribution method '{attribution_method}'. "
                f"Supported methods: {SUPPORTED_STRUCTURED_ATTRIBUTION_METHODS}"
            )

        self.model = model
        self.attribution_method = attribution_method
        self.baseline_type = baseline_type
        self.n_steps = n_steps
        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)

        self._layer_attributors: Dict[str, object] = {}
        self._layer_to_module: Dict[str, nn.Module] = {}
        self._param_to_layer: Dict[str, str] = {}
        self._attribution_cache: Dict[str, torch.Tensor] = {}
        self._build_layer_mapping()

    def _build_layer_mapping(self):
        for name, module in self.model.named_modules():
            self._layer_to_module[name] = module

        for name, _param in self.model.named_parameters():
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                self._param_to_layer[name] = parts[0]

    def _get_layer_module(self, layer_name: str) -> Optional[nn.Module]:
        return self._layer_to_module.get(layer_name)

    def _get_layer_from_param(self, param_name: str) -> Optional[str]:
        return self._param_to_layer.get(param_name)

    def get_layer_from_param(self, param_name: str) -> Optional[str]:
        return self._get_layer_from_param(param_name)

    def _resolve_structured_method(self, method: str) -> str:
        return STRUCTURED_METHOD_ALIASES.get(method, method)

    def _forward_model(self, inputs):
        if isinstance(inputs, dict):
            return self.model(**inputs)
        if isinstance(inputs, (tuple, list)):
            return self.model(*inputs)
        return self.model(inputs)

    def _create_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.baseline_type == "zero":
            return torch.zeros_like(input_tensor)
        if self.baseline_type == "random":
            return torch.randn_like(input_tensor) * 0.01
        if self.baseline_type == "mean":
            return torch.ones_like(input_tensor) * input_tensor.mean()
        return torch.zeros_like(input_tensor)

    def _extract_target_score(
        self,
        outputs,
        target: Union[torch.Tensor, int, None],
    ) -> torch.Tensor:
        if not torch.is_tensor(outputs):
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            else:
                raise ValueError(
                    "Structured attribution currently requires tensor outputs "
                    "or outputs with a .logits tensor."
                )

        if outputs.ndim == 1:
            return outputs.sum()

        if target is None:
            return outputs.max(dim=-1).values.sum()

        if isinstance(target, int):
            return outputs[:, target].sum()

        if not torch.is_tensor(target):
            raise ValueError("Target must be an int or tensor.")

        target = target.to(outputs.device)
        if target.ndim == 0:
            return outputs[:, int(target.item())].sum()

        target = target.long().view(-1, 1)
        if target.shape[0] != outputs.shape[0]:
            raise ValueError(
                f"Target batch size {target.shape[0]} does not match output batch size "
                f"{outputs.shape[0]}."
            )
        return outputs.gather(1, target).sum()

    def _compute_layer_gradient_signal(
        self,
        input_tensor: torch.Tensor,
        target: Union[torch.Tensor, int, None],
        layer_name: str,
        multiply_by_activation: bool = False,
    ) -> Optional[torch.Tensor]:
        layer_module = self._get_layer_module(layer_name)
        if layer_module is None:
            self.logger.warning(f"Layer module not found for {layer_name}")
            return None

        captured = {}

        def capture_activation(_module, _inputs, output):
            captured["activation"] = output[0] if isinstance(output, tuple) else output

        handle = layer_module.register_forward_hook(capture_activation)
        was_training = self.model.training

        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            outputs = self._forward_model(input_tensor)
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError(f"Failed to capture activation for layer '{layer_name}'")

            score = self._extract_target_score(outputs, target)
            gradients = torch.autograd.grad(
                score,
                activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            if multiply_by_activation:
                return activation.detach() * gradients.detach()
            return gradients.detach()
        except Exception as exc:
            self.logger.warning(f"Layer gradient attribution failed for {layer_name}: {exc}")
            return None
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def _get_attributor(self, layer_name: str, method: str):
        resolved_method = self._resolve_structured_method(method)
        cache_key = f"{layer_name}_{resolved_method}"

        if cache_key in self._layer_attributors:
            return self._layer_attributors[cache_key]

        layer_module = self._get_layer_module(layer_name)
        if layer_module is None:
            self.logger.warning(f"Layer module not found for {layer_name}")
            return None

        def forward_func(x):
            return self._forward_model(x)

        try:
            if resolved_method == AttributionMethod.LAYER_CONDUCTANCE.value:
                attributor = LayerConductance(forward_func, layer_module)
            elif resolved_method == AttributionMethod.LAYER_GRADIENT_X_ACTIVATION.value:
                attributor = LayerGradientXActivation(forward_func, layer_module)
            elif resolved_method == AttributionMethod.LAYER_INTEGRATED_GRADIENTS.value:
                attributor = LayerIntegratedGradients(forward_func, layer_module)
            elif resolved_method == AttributionMethod.DEEP_LIFT.value:
                attributor = LayerDeepLift(self.model, layer_module)
            elif resolved_method == "layer_gradient_shap":
                attributor = LayerGradientShap(forward_func, layer_module)
            elif resolved_method == AttributionMethod.LAYER_LRP.value:
                attributor = LayerLRP(self.model, layer_module)
            else:
                return None
        except Exception as exc:
            self.logger.warning(f"Failed to create attributor for {layer_name}: {exc}")
            return None

        self._layer_attributors[cache_key] = attributor
        return attributor

    def compute_layer_attribution(
        self,
        input_tensor: torch.Tensor,
        target: Union[torch.Tensor, int],
        layer_name: str,
    ) -> Optional[torch.Tensor]:
        resolved_method = self._resolve_structured_method(self.attribution_method)

        if resolved_method == "layer_saliency":
            return self._compute_layer_gradient_signal(
                input_tensor=input_tensor,
                target=target,
                layer_name=layer_name,
                multiply_by_activation=False,
            )

        attributor = self._get_attributor(layer_name, self.attribution_method)
        if attributor is None:
            return None

        if isinstance(target, torch.Tensor) and target.dim() > 0 and target.numel() == 1:
            target = target.item()

        baseline = self._create_baseline(input_tensor)
        was_training = self.model.training

        try:
            self.model.eval()
            self.model.zero_grad(set_to_none=True)

            if resolved_method in [
                AttributionMethod.LAYER_CONDUCTANCE.value,
                AttributionMethod.LAYER_INTEGRATED_GRADIENTS.value,
            ]:
                attributions = attributor.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target,
                    n_steps=self.n_steps,
                )
            elif resolved_method == AttributionMethod.DEEP_LIFT.value:
                attributions = attributor.attribute(
                    input_tensor,
                    baselines=baseline,
                    target=target,
                )
            elif resolved_method == AttributionMethod.LAYER_GRADIENT_X_ACTIVATION.value:
                attributions = attributor.attribute(input_tensor, target=target)
            elif resolved_method == "layer_gradient_shap":
                baselines = torch.cat(
                    [self._create_baseline(input_tensor) for _ in range(4)],
                    dim=0,
                )
                attributions = attributor.attribute(
                    input_tensor,
                    baselines=baselines,
                    target=target,
                    n_samples=8,
                )
            elif resolved_method == AttributionMethod.LAYER_LRP.value:
                attributions = attributor.attribute(input_tensor, target=target)
            else:
                raise ValueError(
                    f"Resolved attribution method '{resolved_method}' is not implemented."
                )

            return attributions
        except Exception as exc:
            self.logger.warning(f"Attribution failed for {layer_name}: {exc}")
            return None
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def compute_all_layer_attributions(
        self,
        input_tensor: torch.Tensor,
        target: Union[torch.Tensor, int],
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if layer_names is None:
            layer_names = [
                name
                for name, module in self._layer_to_module.items()
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear))
            ]

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
        if attribution.dim() == 4:
            if aggregation == "sum_abs":
                channel_attr = torch.abs(attribution).sum(dim=(0, 2, 3))
            elif aggregation == "mean_abs":
                channel_attr = torch.abs(attribution).mean(dim=(0, 2, 3))
            else:
                channel_attr = torch.abs(attribution).amax(dim=(0, 2, 3))
        elif attribution.dim() == 3:
            if aggregation == "sum_abs":
                channel_attr = torch.abs(attribution).sum(dim=(1, 2))
            elif aggregation == "mean_abs":
                channel_attr = torch.abs(attribution).mean(dim=(1, 2))
            else:
                channel_attr = torch.abs(attribution).amax(dim=(1, 2))
        elif attribution.dim() == 2:
            if aggregation == "sum_abs":
                channel_attr = torch.abs(attribution).sum(dim=0)
            elif aggregation == "mean_abs":
                channel_attr = torch.abs(attribution).mean(dim=0)
            else:
                channel_attr = torch.abs(attribution).amax(dim=0)
        elif attribution.dim() == 1:
            channel_attr = torch.abs(attribution)
        else:
            channel_attr = torch.abs(attribution).flatten()

        if channel_attr.numel() != num_groups:
            if channel_attr.numel() > num_groups:
                group_size = channel_attr.numel() // num_groups
                if group_size * num_groups == channel_attr.numel():
                    channel_attr = channel_attr[: num_groups * group_size]
                    channel_attr = channel_attr.view(num_groups, group_size).sum(dim=1)
                else:
                    channel_attr = channel_attr[:num_groups]
            else:
                padded = torch.zeros(num_groups, device=channel_attr.device)
                padded[: channel_attr.numel()] = channel_attr
                channel_attr = padded

        return channel_attr

    def update_attribution_cache(
        self,
        layer_name: str,
        new_attribution: torch.Tensor,
        ema_decay: float = 0.9,
    ):
        if layer_name in self._attribution_cache:
            old_attr = self._attribution_cache[layer_name]
            self._attribution_cache[layer_name] = (
                ema_decay * old_attr + (1 - ema_decay) * new_attribution.detach()
            )
        else:
            self._attribution_cache[layer_name] = new_attribution.detach().clone()

    def get_cached_attribution(self, layer_name: str) -> Optional[torch.Tensor]:
        return self._attribution_cache.get(layer_name)

    def clear_cache(self):
        self._attribution_cache.clear()
        self._layer_attributors.clear()
