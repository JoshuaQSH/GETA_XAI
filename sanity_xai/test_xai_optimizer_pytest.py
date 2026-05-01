import pytest
import torch
import torch.nn as nn

from only_train_once import OTO
from only_train_once.optimizer import GETA
from only_train_once.quantization.quant_layers import QuantizationMode
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.xai_optimizer import XAI_GETA
from only_train_once.xai_optimizer.captum_attribution import (
    CaptumAttributionCalculator,
    QUANTIZATION_INCOMPATIBLE_METHODS,
    SUPPORTED_STRUCTURED_ATTRIBUTION_METHODS,
)
from sanity_check.backends.vgg7 import vgg7_bn
import sanity_xai.qvgg7_xai_quickcheck_demo as quickcheck_demo


class TinyConvNet(nn.Module):
    def __init__(self, num_classes=10, inplace=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def create_quantized_model(device, inplace=True):
    model = TinyConvNet(inplace=inplace)
    model = model_to_quantize_model(
        model,
        num_bits=8,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
    )
    return model.to(device)


def create_optimizer(optimizer_cls, model, device, **kwargs):
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    oto = OTO(model, dummy_input)
    param_groups = oto._graph.get_param_groups()
    optimizer = optimizer_cls(
        params=param_groups,
        model=model if optimizer_cls is XAI_GETA else None,
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_periods=2,
        projection_steps=4,
        start_pruning_step=4,
        pruning_periods=2,
        pruning_steps=8,
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=str(device),
        **kwargs,
    )
    return oto, optimizer


def create_geta_optimizer(model, device):
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    oto = OTO(model, dummy_input)
    optimizer = GETA(
        params=oto._graph.get_param_groups(),
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_periods=2,
        projection_steps=4,
        start_pruning_step=4,
        pruning_periods=2,
        pruning_steps=8,
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=str(device),
    )
    return oto, optimizer


def create_batches(device, num_batches=6, batch_size=8):
    torch.manual_seed(7)
    batches = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        batches.append((x, y))
    return batches


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_saliency_is_layer_aware(device):
    model = TinyConvNet(inplace=False).to(device)
    calculator = CaptumAttributionCalculator(
        model=model,
        attribution_method="saliency",
        device=str(device),
    )
    x = torch.randn(2, 3, 16, 16, device=device)
    y = torch.tensor([0, 1], device=device)

    conv1_attr = calculator.compute_layer_attribution(x, y, "conv1")
    conv2_attr = calculator.compute_layer_attribution(x, y, "conv2")

    assert conv1_attr is not None
    assert conv2_attr is not None
    assert conv1_attr.shape == (2, 8, 16, 16)
    assert conv2_attr.shape == (2, 16, 16, 16)
    assert not torch.allclose(conv1_attr[:, :8], conv2_attr[:, :8])


@pytest.mark.parametrize("method", QUANTIZATION_INCOMPATIBLE_METHODS)
def test_unsupported_methods_are_rejected(device, method):
    model = TinyConvNet().to(device)
    with pytest.raises(ValueError):
        CaptumAttributionCalculator(
            model=model,
            attribution_method=method,
            device=str(device),
        )


def test_step_updates_cached_attributions(device):
    torch.manual_seed(11)
    model = create_quantized_model(device, inplace=False)
    _, optimizer = create_optimizer(
        XAI_GETA,
        model,
        device,
        attribution_method="saliency",
        attribution_weight=0.3,
        compute_attribution_freq=1,
        attribution_n_steps=2,
    )

    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    loss = nn.CrossEntropyLoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(inputs=x, targets=y)

    assert optimizer._last_attribution_step == 1
    assert optimizer._cached_attributions
    assert optimizer._attribution_layer_names
    first_layer = optimizer._attribution_layer_names[0]
    assert first_layer in optimizer._cached_attributions
    cached = optimizer._cached_attributions[first_layer]
    assert cached.ndim in (1, 3)


def test_structural_layers_are_preferred_for_group_attribution(device):
    model = vgg7_bn()
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
    ).to(device)
    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    oto = OTO(model, dummy_input)
    optimizer = XAI_GETA(
        params=oto._graph.get_param_groups(),
        model=model,
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_periods=2,
        projection_steps=4,
        start_pruning_step=4,
        pruning_periods=2,
        pruning_steps=8,
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=str(device),
        attribution_method="saliency",
        attribution_weight=0.3,
        compute_attribution_freq=1,
        attribution_n_steps=2,
    )

    first_prunable_group = next(
        group
        for group in optimizer.param_groups
        if group["is_prunable"] and not group["is_auxiliary"]
    )
    assert "features.0" in first_prunable_group["attribution_layer_names"]
    assert "features.1" not in first_prunable_group["attribution_layer_names"]


def test_xai_geta_matches_geta_when_attribution_disabled(device):
    torch.manual_seed(21)
    model_geta = create_quantized_model(device, inplace=False)
    torch.manual_seed(21)
    model_xai = create_quantized_model(device, inplace=False)

    _, geta = create_geta_optimizer(model_geta, device)
    _, xai_geta = create_optimizer(
        XAI_GETA,
        model_xai,
        device,
        attribution_method="saliency",
        attribution_weight=0.0,
        compute_attribution_freq=1000,
        attribution_n_steps=2,
    )

    criterion = nn.CrossEntropyLoss()
    for x, y in create_batches(device, num_batches=8):
        geta.zero_grad()
        loss_geta = criterion(model_geta(x), y)
        loss_geta.backward()
        geta.step()

        xai_geta.zero_grad()
        loss_xai = criterion(model_xai(x), y)
        loss_xai.backward()
        xai_geta.step()

    for (name_geta, param_geta), (name_xai, param_xai) in zip(
        model_geta.named_parameters(), model_xai.named_parameters()
    ):
        assert name_geta == name_xai
        max_diff = (param_geta - param_xai).abs().max().item()
        assert max_diff < 5e-4, f"{name_geta}: max diff {max_diff}"


def test_attribution_modulation_is_bounded(device):
    torch.manual_seed(31)
    model = create_quantized_model(device, inplace=False)
    _, optimizer = create_optimizer(
        XAI_GETA,
        model,
        device,
        attribution_method="saliency",
        attribution_weight=0.3,
        compute_attribution_freq=1,
        attribution_n_steps=2,
    )

    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    loss = nn.CrossEntropyLoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(inputs=x, targets=y)
    optimizer.compute_importance_scores()

    lower = 1.0 - 0.5 * optimizer.attribution_weight
    upper = 1.0 + 0.5 * optimizer.attribution_weight
    bounded_groups = 0
    for group in optimizer.param_groups:
        modulation = group["importance_scores"].get("attribution_modulation")
        if modulation is None:
            continue
        bounded_groups += 1
        assert torch.all(modulation >= lower - 1e-6)
        assert torch.all(modulation <= upper + 1e-6)
    assert bounded_groups > 0


@pytest.mark.parametrize("method", SUPPORTED_STRUCTURED_ATTRIBUTION_METHODS)
def test_supported_methods_complete_stage_smoke(device, method):
    success, error, skipped = quickcheck_demo.test_single_method(
        method_name=method,
        device=device,
        num_epochs=4,
        num_samples=32,
    )

    assert skipped is False
    assert success, error
