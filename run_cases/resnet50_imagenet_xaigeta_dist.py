"""
ResNet50 ImageNet XAI-GETA Experiment — Distributed Multi-GPU Version
=====================================================================

Train ResNet50 on ImageNet using XAI-GETA with PyTorch DistributedDataParallel.

Usage (single node, multiple GPUs):
    # Auto-detect all available GPUs:
    torchrun --nproc_per_node=auto resnet50_imagenet_xaigeta_dist.py --epochs 90

    # Specify exact GPU count:
    torchrun --nproc_per_node=4 resnet50_imagenet_xaigeta_dist.py --epochs 90

    # Quick smoke test (2 GPUs, 2 epochs, small subset):
    torchrun --nproc_per_node=2 resnet50_imagenet_xaigeta_dist.py --test

    # Legacy launcher (deprecated but still works):
    python -m torch.distributed.launch --nproc_per_node=4 resnet50_imagenet_xaigeta_dist.py

DDP Init Order (critical for OTO compatibility):
    1. Create model on local GPU
    2. Initialize OTO (JIT traces the unwrapped model)
    3. Create XAI-GETA optimizer with UNWRAPPED model reference
    4. Compute initial attributions BEFORE DDP wrapping
    5. Wrap model with DistributedDataParallel
    6. Train — optimizer already holds parameter references
"""

import os
import sys
import time
import math
from datetime import datetime, timedelta
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.xai_optimizer import XAI_GETA

from run_cases.utils import (
    get_xai_parser,
    args_to_config,
    ExperimentConfig,
    ExperimentResults,
    save_results_to_csv,
    print_results_summary,
    print_config,
    get_timestamp,
    generate_experiment_name,
    ensure_output_dir,
    accuracy_topk,
    AVAILABLE_METHODS,
)


# ============================================================================
# Distributed Helpers
# ============================================================================

def is_main_process():
    """Check if this is the main (rank 0) process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def dprint(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    """
    Initialize distributed process group.
    Expects environment variables set by torchrun / torch.distributed.launch:
        RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    if "RANK" not in os.environ:
        # Not launched with torchrun — fall back to single-GPU
        print("Warning: RANK env var not set. Running in single-GPU mode.")
        print("Use `torchrun --nproc_per_node=N` for multi-GPU.")
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Distributed: {world_size} GPUs, backend=nccl")

    return rank, local_rank, world_size


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """Average a tensor across all processes."""
    if not dist.is_initialized() or world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ============================================================================
# Dataset
# ============================================================================

class RobustImageFolder(torch.utils.data.Dataset):
    """ImageFolder wrapper that handles corrupted images gracefully."""

    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root, transform=None)
        self.transform = transform
        self.valid_indices = list(range(len(self.dataset)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        path, target = self.dataset.samples[actual_idx]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        except Exception:
            # Return a black image as fallback
            dummy = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform is not None:
                dummy = self.transform(dummy)
            return dummy, 0


def get_imagenet_loaders(config, world_size, rank, max_train_samples=0, max_val_samples=0):
    """Create ImageNet data loaders with DistributedSampler.
    
    Args:
        max_train_samples: If > 0, use only this many training samples (for smoke tests).
        max_val_samples: If > 0, use only this many validation samples.
    """

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trainset = RobustImageFolder(
        root=os.path.join(config.dataset_root, "ImageNet", "train"),
        transform=train_transform,
    )
    testset = RobustImageFolder(
        root=os.path.join(config.dataset_root, "ImageNet", "val"),
        transform=test_transform,
    )

    # Subset for smoke tests
    if max_train_samples > 0 and len(trainset) > max_train_samples:
        trainset = torch.utils.data.Subset(trainset, list(range(max_train_samples)))
    if max_val_samples > 0 and len(testset) > max_val_samples:
        testset = torch.utils.data.Subset(testset, list(range(max_val_samples)))

    # DistributedSampler partitions data across GPUs
    train_sampler = DistributedSampler(
        trainset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None

    test_sampler = DistributedSampler(
        testset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None

    trainloader = DataLoader(
        trainset,
        batch_size=config.batch_size,  # per-GPU batch size
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return trainloader, testloader, trainset, train_sampler


# ============================================================================
# Model
# ============================================================================

def create_quantized_resnet50(device):
    """Create quantized ResNet50 model and dummy input."""
    import torchvision.models as models
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode

    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model = model_to_quantize_model(
        model,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
        q_m_init=1.0,
    )
    dummy_input = torch.rand(1, 3, 224, 224)
    return model.to(device), dummy_input.to(device)


# ============================================================================
# Evaluation (distributed-aware)
# ============================================================================

def evaluate_model_dist(model, testloader, device, world_size):
    """Evaluate model accuracy, aggregating across all GPUs."""
    model.eval()
    local_correct1 = torch.tensor(0.0, device=device)
    local_correct5 = torch.tensor(0.0, device=device)
    local_total = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            local_total += y.size(0)
            local_correct1 += prec1.item() * y.size(0) / 100.0
            local_correct5 += prec5.item() * y.size(0) / 100.0

    # Sum across all ranks
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(local_correct1, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_correct5, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)

    acc1 = (local_correct1 / local_total * 100.0).item() if local_total > 0 else 0.0
    acc5 = (local_correct5 / local_total * 100.0).item() if local_total > 0 else 0.0
    model.train()
    return acc1, acc5


# ============================================================================
# Training
# ============================================================================

def train_xai_geta_dist(
    trainloader,
    testloader,
    trainset,
    train_sampler,
    model,
    ddp_model,
    oto,
    optimizer,
    config,
    rank,
    local_rank,
    world_size,
):
    """
    Distributed training loop for ResNet50 with XAI-GETA.

    Args:
        model: Unwrapped model (for OTO/metrics)
        ddp_model: DDP-wrapped model (for forward/backward)
        optimizer: XAI_GETA optimizer (created with unwrapped model)
    """
    device = torch.device(f"cuda:{local_rank}")
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )

    # Effective batch size for logging
    effective_bs = config.batch_size * world_size
    dprint(f"\nEffective batch size: {config.batch_size} x {world_size} GPUs = {effective_bs}")

    # Initial MACs/BOPs (only rank 0 needs this)
    full_macs = full_bops = 0.0
    if rank == 0:
        dprint("Computing initial MACs and BOPs...")
        full_macs = oto.compute_macs(in_million=True)["total"]
        full_bops = oto.compute_bops(in_million=True)["total"]
        dprint(f"Full MACs: {full_macs:.2f} M")
        dprint(f"Full BOPs: {full_bops:.2f} M")

    dprint("\nStarting Distributed XAI-GETA Training...")
    dprint("-" * 70)

    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        # Shuffle data differently each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        ddp_model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        num_batches = torch.tensor(0.0, device=device)

        for batch_idx, (X, y) in enumerate(trainloader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass through DDP model
            y_pred = ddp_model(X)
            loss = criterion(y_pred, y)

            # Backward pass — DDP synchronizes gradients automatically
            optimizer.zero_grad()
            loss.backward()

            epoch_loss += loss.detach()
            num_batches += 1

            # XAI-GETA optimizer step (projection/pruning/quantization)
            optimizer.step(inputs=X, targets=y)

        lr_scheduler.step()

        # Aggregate loss across GPUs
        if world_size > 1:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        avg_loss = (epoch_loss / num_batches).item()

        # Evaluate (all ranks participate, result aggregated)
        acc1, acc5 = evaluate_model_dist(ddp_model, testloader, device, world_size)
        metrics = optimizer.compute_metrics()

        if rank == 0:
            print(
                f"Ep: {epoch:3d}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc@1: {acc1:.2f}% | "
                f"Acc@5: {acc5:.2f}% | "
                f"GrpSparsity: {metrics.group_sparsity:.4f} | "
                f"NormParams: {metrics.norm_params:.2f} | "
                f"#Imp: {metrics.num_important_groups} | "
                f"#Red: {metrics.num_redundant_groups}"
            )

    total_time = time.time() - start_time

    # Final eval and results (rank 0 only)
    final_acc1, final_acc5 = evaluate_model_dist(ddp_model, testloader, device, world_size)

    results = None
    if rank == 0:
        final_metrics = optimizer.compute_metrics()
        compressed_macs = oto.compute_macs(in_million=True)["total"]
        compressed_bops = oto.compute_bops(in_million=True)["total"]
        dprint(f"\nCompressed MACs: {compressed_macs:.2f} M")
        dprint(f"Compressed BOPs: {compressed_bops:.2f} M")

        results = ExperimentResults(
            model_name="ResNet50",
            optimizer_type="XAI-GETA-DDP",
            epochs=config.epochs,
            target_sparsity=config.target_sparsity,
            attribution_method=config.attribution_method,
            attribution_weight=config.attribution_weight,
            full_macs=full_macs,
            full_bops=full_bops,
            compressed_macs=compressed_macs,
            compressed_bops=compressed_bops,
            final_top1_accuracy=final_acc1,
            final_top5_accuracy=final_acc5,
            total_param_norm=final_metrics.norm_params,
            group_sparsity=final_metrics.group_sparsity,
            num_important_groups=final_metrics.num_important_groups,
            num_redundant_groups=final_metrics.num_redundant_groups,
            num_zero_groups=final_metrics.num_zero_groups,
            total_training_time=total_time,
            timestamp=get_timestamp(),
        )

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    # ---- Parse arguments ----
    parser = get_xai_parser("ResNet50 ImageNet XAI-GETA (Distributed)")
    parser.add_argument(
        "--test", action="store_true",
        help="Quick smoke test: 2 epochs, small data subset",
    )
    args = parser.parse_args()
    config = args_to_config(args, is_xai=True)

    # ---- Distributed setup ----
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Apply smoke-test overrides
    if args.test:
        config.epochs = 2
        config.projection_epochs = 1
        config.pruning_epochs = 1
        config.projection_periods = 1
        config.pruning_periods = 1
        config.num_workers = 2  # Fewer workers for faster startup
        dprint("[SMOKE TEST] Running 2 epochs with minimal config...")

    if config.attribution_method not in AVAILABLE_METHODS:
        dprint(f"Error: Unknown attribution method '{config.attribution_method}'")
        cleanup_distributed()
        sys.exit(1)

    dprint("\n" + "=" * 70)
    dprint(f"ResNet50 ImageNet XAI-GETA — Distributed ({world_size} GPUs)")
    dprint("=" * 70)
    if rank == 0:
        print_config(config, "XAI-GETA-DDP")
    ensure_output_dir(config)

    # ---- Step 1: Create model (BEFORE DDP) ----
    dprint("[Step 1] Creating quantized ResNet50...")
    model, dummy_input = create_quantized_resnet50(device)

    # ---- Step 2: Init OTO on UNWRAPPED model (does JIT tracing) ----
    dprint("[Step 2] Initializing OTO framework...")
    oto = OTO(model=model, dummy_input=dummy_input)

    # ---- Step 3: Create XAI-GETA optimizer with UNWRAPPED model ----
    dprint("[Step 3] Creating XAI-GETA optimizer...")
    if args.test:
        # Use tiny dataset size for step calculations in test mode
        num_train_samples = 512
    else:
        num_train_samples = 1281167  # ImageNet ~1.28M images
    steps_per_epoch = num_train_samples // (config.batch_size * world_size)
    projection_steps = config.projection_epochs * steps_per_epoch
    pruning_steps = config.pruning_epochs * steps_per_epoch
    start_pruning_step = projection_steps

    param_groups = oto._graph.get_param_groups()
    optimizer = XAI_GETA(
        params=param_groups,
        model=model,  # UNWRAPPED model for Captum
        variant="sgd",
        lr=config.lr,
        lr_quant=config.lr_quant,
        first_momentum=0.9,
        weight_decay=config.weight_decay,
        target_group_sparsity=config.target_sparsity,
        start_projection_step=config.start_projection_step,
        projection_periods=config.projection_periods,
        projection_steps=projection_steps,
        start_pruning_step=start_pruning_step,
        pruning_periods=config.pruning_periods,
        pruning_steps=pruning_steps,
        bit_reduction=config.bit_reduction,
        min_bit_wt=config.min_bit,
        max_bit_wt=config.max_bit,
        device=str(device),
        attribution_method=config.attribution_method,
        attribution_weight=config.attribution_weight,
        ema_decay=config.ema_decay,
        compute_attribution_freq=config.update_freq,
        attribution_n_steps=config.attribution_n_steps,
    )
    oto._optimizer = optimizer

    dprint(f"  Projection steps: {projection_steps} ({config.projection_epochs} ep)")
    dprint(f"  Pruning steps:    {pruning_steps} ({config.pruning_epochs} ep)")
    dprint(f"  Steps/epoch:      ~{steps_per_epoch}")

    # ---- Step 4: Compute initial attributions (BEFORE DDP) ----
    if args.test:
        dprint("[Step 4] Skipping initial attributions (smoke test — magnitude-only fallback)")
    else:
        dprint("[Step 4] Computing initial attributions...")
        try:
            small_loader = DataLoader(
                RobustImageFolder(
                    root=os.path.join(config.dataset_root, "ImageNet", "train"),
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]),
                ),
                batch_size=8,
                shuffle=True,
                num_workers=2,
            )
            optimizer.compute_initial_attributions(small_loader, num_batches=2)
            dprint(f"  Attributions computed for {len(optimizer._cached_attributions)} layers")
            del small_loader
            torch.cuda.empty_cache()
        except Exception as e:
            dprint(f"  Warning: Initial attributions failed: {e}")
            dprint(f"  Continuing with fallback importance scoring...")

    # ---- Step 5: Wrap model with DDP ----
    dprint("[Step 5] Wrapping model with DistributedDataParallel...")
    if world_size > 1:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    # ---- Step 6: Load dataset with DistributedSampler ----
    dprint("[Step 6] Loading ImageNet dataset...")
    max_train = 512 if args.test else 0  # Small subset for smoke test
    max_val = 256 if args.test else 0
    trainloader, testloader, trainset, train_sampler = get_imagenet_loaders(
        config, world_size, rank, max_train_samples=max_train, max_val_samples=max_val
    )
    dprint(f"  Training samples: {len(trainset)} (each GPU sees ~{len(trainset) // world_size})")
    dprint(f"  Val samples: {len(testloader.dataset)}")

    # Re-compute steps_per_epoch from actual loader
    steps_per_epoch = len(trainloader)
    dprint(f"  Actual steps/epoch: {steps_per_epoch}")

    # ---- Step 7: Train ----
    dprint(f"\n[Step 7] Training with XAI-GETA ({world_size} GPUs)...")
    dprint(f"  Method: {config.attribution_method}, Weight: {config.attribution_weight}")

    results = train_xai_geta_dist(
        trainloader, testloader, trainset, train_sampler,
        model, ddp_model, oto, optimizer, config,
        rank, local_rank, world_size,
    )

    # ---- Save results (rank 0 only) ----
    if rank == 0 and results is not None:
        print_results_summary(results)
        exp_name = generate_experiment_name(
            "xai_geta_dist", config, model_name="resnet50_imagenet"
        )
        csv_path = os.path.join(config.output_dir, f"{exp_name}_results.csv")
        save_results_to_csv(results, csv_path)
        dprint("\n" + "=" * 70)
        dprint("XAI-GETA Distributed Experiment Complete!")
        dprint("=" * 70 + "\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
