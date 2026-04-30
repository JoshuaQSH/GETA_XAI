"""
Sanity Check: Compare GETA vs XAI_GETA (with attribution_weight=0)
===================================================================

When attribution_weight=0, XAI_GETA should produce identical results to GETA.
This script verifies that behavior using synthetic data for fast iteration.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.optimizer import GETA
from only_train_once.xai_optimizer import XAI_GETA


def create_simple_model():
    """Create a simple quantized model for testing."""
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)
            
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = SimpleNet()
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION,
        q_m_init=1.0,
    )
    return model


def create_synthetic_data(batch_size=32, num_batches=10):
    """Create synthetic data for testing."""
    data = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        data.append((x, y))
    return data


def setup_geta(model, device):
    """Setup GETA optimizer."""
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model, dummy_input)
    
    param_groups = oto._graph.get_param_groups()
    
    optimizer = GETA(
        params=param_groups,
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_periods=2,
        projection_steps=100,  # 10 steps per period
        start_pruning_step=100,
        pruning_periods=5,
        pruning_steps=200,  # 40 steps per period
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=device,
    )
    
    return oto, optimizer


def setup_xai_geta(model, device, attribution_weight=0.0):
    """Setup XAI_GETA optimizer with specified attribution weight."""
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    oto = OTO(model, dummy_input)
    
    param_groups = oto._graph.get_param_groups()
    
    optimizer = XAI_GETA(
        params=param_groups,
        model=model,
        variant="adam",
        lr=1e-3,
        lr_quant=1e-3,
        first_momentum=0.9,
        weight_decay=1e-4,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_periods=2,
        projection_steps=100,
        start_pruning_step=100,
        pruning_periods=5,
        pruning_steps=200,
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=device,
        attribution_method="saliency",
        attribution_weight=attribution_weight,
        compute_attribution_freq=10000,  # Don't compute attributions
    )
    
    return oto, optimizer


def train_one_epoch(model, optimizer, data, criterion, device):
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0
    
    for x, y in data:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    metrics = optimizer.compute_metrics()
    
    return total_loss / len(data), metrics


def get_model_state(model):
    """Get a summary of model state for comparison."""
    state = {}
    for name, param in model.named_parameters():
        if param.numel() > 0:
            state[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item() if param.numel() > 1 else 0,
                'norm': param.data.norm().item(),
                'zeros': (param.data == 0).sum().item(),
                'total': param.numel(),
            }
    return state


def compare_states(state1, state2, tolerance=1e-5):
    """Compare two model states and report differences."""
    differences = []
    
    for name in state1.keys():
        if name not in state2:
            differences.append(f"  {name}: missing in state2")
            continue
        
        s1, s2 = state1[name], state2[name]
        for key in ['mean', 'std', 'norm']:
            diff = abs(s1[key] - s2[key])
            if diff > tolerance:
                differences.append(f"  {name}.{key}: {s1[key]:.6f} vs {s2[key]:.6f} (diff: {diff:.6f})")
    
    return differences


def run_sanity_check():
    """Run the sanity check comparing GETA vs XAI_GETA."""
    print("=" * 70)
    print("Sanity Check: GETA vs XAI_GETA (attribution_weight=0)")
    print("=" * 70)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    print("\n[1] Creating synthetic data...")
    data = create_synthetic_data()
    
    # Create two identical models
    print("[2] Creating models...")
    torch.manual_seed(42)
    model_geta = create_simple_model().to(device)
    
    torch.manual_seed(42)
    model_xai = create_simple_model().to(device)
    
    # Verify models start identical
    state_geta_init = get_model_state(model_geta)
    state_xai_init = get_model_state(model_xai)
    init_diff = compare_states(state_geta_init, state_xai_init)
    if init_diff:
        print("WARNING: Initial models are different!")
        for d in init_diff[:5]:
            print(d)
    else:
        print("✓ Initial models are identical")
    
    # Setup optimizers
    print("\n[3] Setting up optimizers...")
    torch.manual_seed(42)
    oto_geta, opt_geta = setup_geta(model_geta, device)
    
    torch.manual_seed(42)
    oto_xai, opt_xai = setup_xai_geta(model_xai, device, attribution_weight=0.0)
    
    # Compare importance score criteria
    print("\nGETA importance criteria:", opt_geta.importance_score_criteria)
    print("XAI_GETA importance criteria:", opt_xai.importance_score_criteria)
    
    if opt_geta.importance_score_criteria != opt_xai.importance_score_criteria:
        print("❌ ISSUE: Importance score criteria are different!")
    else:
        print("✓ Importance score criteria are identical")
    
    # Train and compare
    print("\n[4] Training and comparing...")
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 50  # Enough to go through projection + some pruning
    
    for epoch in range(1, num_epochs + 1):
        # Reset data seed for each epoch to ensure same data order
        torch.manual_seed(42 + epoch)
        
        loss_geta, metrics_geta = train_one_epoch(model_geta, opt_geta, data, criterion, device)
        
        torch.manual_seed(42 + epoch)
        loss_xai, metrics_xai = train_one_epoch(model_xai, opt_xai, data, criterion, device)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}:")
            print(f"  GETA     - Loss: {loss_geta:.4f}, Sparsity: {metrics_geta.group_sparsity:.4f}, NormParams: {metrics_geta.norm_params:.2f}")
            print(f"  XAI_GETA - Loss: {loss_xai:.4f}, Sparsity: {metrics_xai.group_sparsity:.4f}, NormParams: {metrics_xai.norm_params:.2f}")
            
            loss_diff = abs(loss_geta - loss_xai)
            sparsity_diff = abs(metrics_geta.group_sparsity - metrics_xai.group_sparsity)
            norm_diff = abs(metrics_geta.norm_params - metrics_xai.norm_params)
            
            if loss_diff > 0.01 or sparsity_diff > 0.01 or norm_diff > 1.0:
                print(f"  ⚠️ DIVERGENCE DETECTED!")
                print(f"     Loss diff: {loss_diff:.6f}")
                print(f"     Sparsity diff: {sparsity_diff:.6f}")
                print(f"     NormParams diff: {norm_diff:.2f}")
    
    # Final comparison
    print("\n[5] Final model state comparison...")
    state_geta_final = get_model_state(model_geta)
    state_xai_final = get_model_state(model_xai)
    final_diff = compare_states(state_geta_final, state_xai_final, tolerance=0.1)
    
    if final_diff:
        print("❌ Final models are DIFFERENT:")
        for d in final_diff[:10]:
            print(d)
    else:
        print("✓ Final models are identical (within tolerance)")
    
    print("\n" + "=" * 70)
    print("Sanity Check Complete")
    print("=" * 70)


if __name__ == "__main__":
    run_sanity_check()
