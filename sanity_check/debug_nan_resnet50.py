"""
Debug script to trace NaN source in ResNet50+ImageNet with XAI_GETA
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.xai_optimizer import XAI_GETA, CaptumAttributionCalculator
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


def check_nan(name, tensor):
    """Check if tensor has NaN and print details."""
    if tensor is None:
        return False
    if isinstance(tensor, (list, tuple)):
        for i, t in enumerate(tensor):
            check_nan(f"{name}[{i}]", t)
        return
    if not isinstance(tensor, torch.Tensor):
        return False
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}: shape={tensor.shape}, max={tensor.max()}, min={tensor.min()}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}: shape={tensor.shape}, max={tensor.max()}, min={tensor.min()}")
        return True
    return False


def register_hooks(model):
    """Register forward/backward hooks to detect NaN."""
    nan_detected = {'forward': None, 'backward': None}
    
    def make_forward_hook(name):
        def hook(module, input, output):
            # Check input
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if check_nan(f"FWD {name} input[{i}]", inp):
                        nan_detected['forward'] = name
            elif check_nan(f"FWD {name} input", input):
                nan_detected['forward'] = name
            
            # Check output
            if isinstance(output, tuple):
                for i, out in enumerate(output):
                    if check_nan(f"FWD {name} output[{i}]", out):
                        nan_detected['forward'] = name
            elif check_nan(f"FWD {name} output", output):
                nan_detected['forward'] = name
        return hook
    
    def make_backward_hook(name):
        def hook(module, grad_input, grad_output):
            # Check grad_output
            if isinstance(grad_output, tuple):
                for i, g in enumerate(grad_output):
                    if check_nan(f"BWD {name} grad_output[{i}]", g):
                        nan_detected['backward'] = name
            elif check_nan(f"BWD {name} grad_output", grad_output):
                nan_detected['backward'] = name
            
            # Check grad_input
            if isinstance(grad_input, tuple):
                for i, g in enumerate(grad_input):
                    if check_nan(f"BWD {name} grad_input[{i}]", g):
                        nan_detected['backward'] = name
            elif check_nan(f"BWD {name} grad_input", grad_input):
                nan_detected['backward'] = name
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            hooks.append(module.register_forward_hook(make_forward_hook(name)))
            hooks.append(module.register_full_backward_hook(make_backward_hook(name)))
    
    return hooks, nan_detected


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model (use WEIGHT_ONLY mode - activation quantization not fully supported in XAI_GETA)
    print("\n[1] Creating model...")
    model = models.resnet50(pretrained=True)
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_ONLY,  # Only weight quantization is supported
        q_m_init=1.0,  # Use default q_m to avoid overflow
    )
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    
    # Setup OTO
    print("\n[2] Setting up OTO...")
    oto = OTO(model, dummy_input)
    
    # Setup optimizer
    print("\n[3] Setting up XAI_GETA optimizer...")
    
    # Short training config
    steps_per_epoch = 100
    projection_epochs = 2
    pruning_epochs = 3
    
    param_groups = oto._graph.get_param_groups()
    
    optimizer = XAI_GETA(
        params=param_groups,
        model=model,
        variant="sgd",
        lr=1e-3,
        first_momentum=0.9,
        target_group_sparsity=0.5,
        start_projection_step=0,
        projection_steps=steps_per_epoch * projection_epochs,
        projection_periods=2,
        start_pruning_step=steps_per_epoch * projection_epochs,
        pruning_steps=steps_per_epoch * pruning_epochs,
        pruning_periods=3,
        bit_reduction=2,
        min_bit_wt=4,
        max_bit_wt=16,
        device=device,
        attribution_method='saliency',
        attribution_weight=0.0,
        compute_attribution_freq=50,
    )
    
    # Enable anomaly detection instead of hooks
    print("\n[4] Enabling autograd anomaly detection...")
    torch.autograd.set_detect_anomaly(True)
    
    # Load a few ImageNet samples
    print("\n[5] Loading data...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = '../datasets/ImageNet/val'
    if os.path.exists(dataset_path):
        dataset = ImageFolder(root=dataset_path, transform=transform)
        # Just take a small subset
        dataset = torch.utils.data.Subset(dataset, range(min(100, len(dataset))))
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    else:
        print(f"Dataset not found at {dataset_path}, using synthetic data")
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return torch.randn(3, 224, 224), torch.randint(0, 1000, (1,)).item()
        loader = DataLoader(SyntheticDataset(), batch_size=8, shuffle=True)
    
    # Training loop
    print("\n[6] Training with NaN detection...")
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        
        print(f"\nStep {step}:")
        
        # Check input
        if check_nan("Input images", images):
            print("ERROR: Input has NaN!")
            break
        
        # Check quant params BEFORE forward
        bad_quant_layers = []
        for name, module in model.named_modules():
            try:
                if hasattr(module, 'd_quant_wt'):
                    d = module.d_quant_wt.item()
                    qm = module.q_m_wt.item()
                    if d <= 0 or math.isnan(d) or math.isinf(d) or d < 1e-10:
                        bad_quant_layers.append(f"{name}:d={d}")
                    if qm <= 0 or math.isnan(qm) or math.isinf(qm):
                        bad_quant_layers.append(f"{name}:qm={qm}")
                if hasattr(module, 'd_quant_act'):
                    d = module.d_quant_act.item()
                    qm = module.q_m_act.item()
                    if d <= 0 or math.isnan(d) or math.isinf(d) or d < 1e-10:
                        bad_quant_layers.append(f"{name}:d_act={d}")
                    if qm <= 0 or math.isnan(qm) or math.isinf(qm):
                        bad_quant_layers.append(f"{name}:qm_act={qm}")
            except Exception as e:
                bad_quant_layers.append(f"{name}:error={e}")
        if bad_quant_layers:
            print(f"  BAD quant params (BEFORE forward): {bad_quant_layers[:10]}")
        
        # Forward
        optimizer.zero_grad()
        try:
            outputs = model(images)
        except Exception as e:
            print(f"ERROR in forward: {e}")
            break
        
        if check_nan("Outputs", outputs):
            print(f"ERROR: Forward produced NaN at step {step}")
            break
        
        # Loss
        loss = criterion(outputs, targets)
        if torch.isnan(loss):
            print(f"ERROR: Loss is NaN at step {step}")
            break
        
        print(f"  Loss: {loss.item():.4f}")
        
        # Check quantization params before backward
        for name, module in model.named_modules():
            if hasattr(module, 'd_quant_wt'):
                d = module.d_quant_wt.item()
                qm = module.q_m_wt.item()
                if d <= 0 or math.isnan(d) or math.isinf(d):
                    print(f"    BAD d_quant_wt in {name}: {d}")
                if qm <= 0 or math.isnan(qm) or math.isinf(qm):
                    print(f"    BAD q_m_wt in {name}: {qm}")
        
        # Backward
        try:
            loss.backward()
        except Exception as e:
            print(f"ERROR in backward: {e}")
            break
        
        # Check gradients
        nan_grad_params = []
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grad_params.append(name)
        
        if nan_grad_params:
            print(f"ERROR: NaN gradients in: {nan_grad_params[:5]}...")
            break
        
        # Step
        optimizer.step()
        
        # Check params after step
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
        
        if nan_params:
            print(f"ERROR: NaN params after step: {nan_params[:5]}...")
            break
        
        print(f"  Step completed OK")
        
        if step >= 50:
            print("\n50 steps completed without NaN!")
            break
    
    print("\n[DONE]")


if __name__ == '__main__':
    main()
