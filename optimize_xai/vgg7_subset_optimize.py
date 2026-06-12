import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from only_train_once import OTO
from only_train_once.optimizer import GETA
from run_cases.utils import (
    create_quantized_resnet18,
    create_quantized_resnet20,
    create_quantized_vgg7,
    evaluate_model,
)


@dataclass
class EpochMetric:
    epoch: int
    loss: float
    acc1: float
    acc5: float
    group_sparsity: float
    num_important_groups: int
    num_redundant_groups: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Small-subset VGG7 optimization runner for XAI-GETA variants."
    )
    parser.add_argument(
        "--optimizer",
        choices=["geta", "xai", "xai_v2", "xai_v3", "xai_v4", "xai_v5", "xai_v6", "xai_v7", "xai_v8", "xai_v9", "xai_v10", "xai_v11", "xai_v12", "xai_v13", "xai_v14", "xai_v15", "xai_v16", "xai_v17", "xai_v18", "xai_v19", "xai_v20", "xai_v21", "xai_v22", "xai_v23", "xai_v24", "xai_v25", "xai_v26", "xai_v27", "xai_v28", "xai_v29", "xai_v30", "xai_v31", "xai_v32", "xai_v33", "xai_v34"],
        default="geta",
        help="Optimizer family to run.",
    )
    parser.add_argument(
        "--model",
        choices=["vgg7", "resnet20", "resnet18"],
        default="vgg7",
        help="Quantized CIFAR-10 model backend to use.",
    )
    parser.add_argument("--method", default="saliency", help="Attribution method.")
    parser.add_argument("--weight", type=float, default=0.3, help="Attribution weight.")
    parser.add_argument(
        "--committee-methods",
        default="saliency,deep_lift",
        help="Comma-separated committee methods for xai_v3.",
    )
    parser.add_argument(
        "--committee-weights",
        default="0.7,0.3",
        help="Comma-separated committee weights for xai_v3.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=384,
        help="Number of training samples in the fixed subset.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of evaluation samples in the fixed subset.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--projection-epochs", type=int, default=4)
    parser.add_argument("--pruning-epochs", type=int, default=6)
    parser.add_argument("--projection-periods", type=int, default=2)
    parser.add_argument("--pruning-periods", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-quant", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        choices=["step", "multistep", "cosine", "none"],
        default="step",
    )
    parser.add_argument("--scheduler-step-size", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument(
        "--scheduler-milestones",
        default="35,50",
        help="Comma-separated epoch milestones for multistep schedule.",
    )
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--bit-reduction", type=int, default=2)
    parser.add_argument("--min-bit", type=int, default=4)
    parser.add_argument("--max-bit", type=int, default=16)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--attribution-freq", type=int, default=8)
    parser.add_argument("--attribution-n-steps", type=int, default=5)
    parser.add_argument("--protection-quantile", type=float, default=0.6)
    parser.add_argument("--progressive-power", type=float, default=1.0)
    parser.add_argument("--max-protection-boost", type=float, default=1.0)
    parser.add_argument(
        "--phase-method-weights",
        default="0.90,0.10;0.65,0.35;0.30,0.70",
        help="Semicolon-separated committee weights per pruning phase for xai_v4.",
    )
    parser.add_argument(
        "--phase-attribution-weights",
        default="0.60,0.45,0.20",
        help="Comma-separated attribution weights per pruning phase for xai_v4.",
    )
    parser.add_argument(
        "--xai-switch-period",
        type=int,
        default=-1,
        help="Pruning period where tapering variants start decaying toward pure GETA scoring.",
    )
    parser.add_argument(
        "--xai-decay-periods",
        type=int,
        default=2,
        help="Number of pruning periods tapering variants use to decay attribution guidance to zero.",
    )
    parser.add_argument(
        "--modulation-clip-start",
        type=float,
        default=0.35,
        help="Initial relative-attribution clamp radius for xai_v8.",
    )
    parser.add_argument(
        "--modulation-clip-end",
        type=float,
        default=0.10,
        help="Final relative-attribution clamp radius for xai_v8.",
    )
    parser.add_argument(
        "--late-protection-scale",
        type=float,
        default=0.5,
        help="Scale factor for the late one-sided protection phase in xai_v9.",
    )
    parser.add_argument(
        "--rescue-fraction",
        type=float,
        default=0.25,
        help="Maximum fraction of late prune candidates xai_v10 may rescue.",
    )
    parser.add_argument(
        "--rescue-pool-factor",
        type=float,
        default=3.0,
        help="Candidate-pool expansion factor for xai_v10 late rescue selection.",
    )
    parser.add_argument(
        "--quant-sensitivity-weight",
        type=float,
        default=0.2,
        help="Strength of the custom quantization-fragility boost in xai_v12.",
    )
    parser.add_argument(
        "--quant-sensitivity-start-period",
        type=int,
        default=-1,
        help="Pruning period where xai_v12 starts applying quantization-fragility boosting.",
    )
    parser.add_argument(
        "--stability-ema-decay",
        type=float,
        default=0.97,
        help="EMA decay for the post-pruning stability anchor in xai_v13.",
    )
    parser.add_argument(
        "--stability-blend",
        type=float,
        default=0.05,
        help="Blend factor for pulling parameters toward the post-pruning EMA anchor in xai_v13.",
    )
    parser.add_argument(
        "--quant-focus-quantile",
        type=float,
        default=0.8,
        help="Top-quantile cutoff for focused quant-sensitivity boosting in xai_v14.",
    )
    parser.add_argument(
        "--quant-focus-start-quantile",
        type=float,
        default=-1.0,
        help="Optional starting focus quantile for scheduled focused boosting in xai_v16.",
    )
    parser.add_argument(
        "--score-ema-decay",
        type=float,
        default=0.8,
        help="EMA decay for pruning-score smoothing in xai_v19.",
    )
    parser.add_argument(
        "--score-ema-start-period",
        type=int,
        default=-1,
        help="Pruning period where xai_v19 starts smoothing prune ranks.",
    )
    parser.add_argument(
        "--score-collection-interval",
        type=int,
        default=1,
        help="Number of optimizer steps between score samples for xai_v21.",
    )
    parser.add_argument(
        "--score-collection-start-period",
        type=int,
        default=-1,
        help="Pruning period where xai_v21 starts accumulating period scores.",
    )
    parser.add_argument(
        "--boundary-pool-factor",
        type=float,
        default=2.0,
        help="Candidate-pool expansion factor for xai_v22 boundary reranking.",
    )
    parser.add_argument(
        "--boundary-mix-weight",
        type=float,
        default=0.35,
        help="Strength of focused boundary reranking in xai_v22.",
    )
    parser.add_argument(
        "--boundary-margin-power",
        type=float,
        default=2.0,
        help="Power applied to prune-boundary proximity in xai_v22.",
    )
    parser.add_argument(
        "--final-eval-ema-decay",
        type=float,
        default=0.995,
        help="EMA decay for late evaluation-only averaging in xai_v26.",
    )
    parser.add_argument(
        "--final-eval-start-epoch",
        type=int,
        default=-1,
        help="Epoch where xai_v26 starts tracking the evaluation-only EMA tail.",
    )
    parser.add_argument(
        "--quant-sensitivity-power",
        type=float,
        default=2.0,
        help="Power for the delayed quant-sensitivity ramp in xai_v27.",
    )
    parser.add_argument(
        "--gamma-modulation-strength",
        type=float,
        default=0.3,
        help="Strength of focus-aware gamma modulation in xai_v28.",
    )
    parser.add_argument(
        "--recovery-lr-boost",
        type=float,
        default=0.3,
        help="Peak post-pruning LR boost in xai_v30.",
    )
    parser.add_argument(
        "--recovery-quant-boost",
        type=float,
        default=0.1,
        help="Peak post-pruning quant LR boost in xai_v30.",
    )
    parser.add_argument(
        "--max-focus-prune-fraction",
        type=float,
        default=0.0,
        help="Maximum fraction of each prune step that xai_v31 may take from V14's focused slice.",
    )
    parser.add_argument(
        "--d-quant-focus-blend",
        type=float,
        default=0.5,
        help="Strength of focus-aware d_quant protection in xai_v32.",
    )
    parser.add_argument(
        "--handoff-anchor-blend",
        type=float,
        default=0.2,
        help="Peak focus-scaled pull toward the pruning-completion snapshot in xai_v33.",
    )
    parser.add_argument(
        "--handoff-anchor-epochs",
        type=int,
        default=12,
        help="Number of post-pruning epochs over which xai_v33 releases its handoff anchor.",
    )
    parser.add_argument(
        "--tail-proxy-drift-weight",
        type=float,
        default=0.1,
        help="Weight of handoff-basin drift in xai_v34's post-pruning checkpoint selector.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset-root", default="../datasets")
    parser.add_argument("--output-dir", default="./outputs/update_optimize")
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--log-epochs",
        default="1,2,3,4,5,6,8,10,12,15,20,25,30",
        help="Comma-separated epochs to print during training.",
    )
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    if args.optimizer == "geta":
        return f"geta_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
    if args.optimizer == "xai_v3":
        committee_tag = "-".join(args.committee_methods.split(","))
        return (
            f"{args.optimizer}_{committee_tag}_w{args.weight}_s{args.sparsity}"
            f"_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer in {"xai_v5", "xai_v6", "xai_v7", "xai_v8"}:
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_dc{args.xai_decay_periods}_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v9":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        late_tag = int(round(args.late_protection_scale * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_lp{late_tag}_pq{int(round(args.protection_quantile * 100))}"
            f"_mb{int(round(args.max_protection_boost * 10))}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v10":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        rescue_tag = int(round(args.rescue_fraction * 100))
        pool_tag = int(round(args.rescue_pool_factor * 10))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_rq{int(round(args.protection_quantile * 100))}_rf{rescue_tag}"
            f"_pf{pool_tag}_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v11":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v12":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_s{args.sparsity}"
            f"_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v13":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        ema_tag = int(round(args.stability_ema_decay * 100))
        blend_tag = int(round(args.stability_blend * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_ed{ema_tag}_sb{blend_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v14":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v15":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        rescue_tag = int(round(args.rescue_fraction * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_rf{rescue_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v16":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_end_tag = int(round(args.quant_focus_quantile * 100))
        focus_start = (
            args.quant_focus_quantile
            if args.quant_focus_start_quantile < 0
            else args.quant_focus_start_quantile
        )
        focus_start_tag = int(round(focus_start * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fs{focus_start_tag}_fe{focus_end_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v17":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        phase_specs = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_specs.append(weights)
        if not phase_specs:
            phase_specs = [[float(w) for w in args.committee_weights.split(",") if w.strip()]]
        start_committee_tag = int(round(phase_specs[0][0] * 100))
        end_committee_tag = int(round(phase_specs[-1][0] * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_cm{start_committee_tag}-{end_committee_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v18":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        phase_specs = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_specs.append(weights)
        if not phase_specs:
            phase_specs = [[float(w) for w in args.committee_weights.split(",") if w.strip()]]
        start_committee_tag = int(round(phase_specs[0][0] * 100))
        end_committee_tag = int(round(phase_specs[-1][0] * 100))
        blend_tag = int(round(args.stability_blend * 100))
        ema_tag = int(round(args.stability_ema_decay * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_cm{start_committee_tag}-{end_committee_tag}"
            f"_ed{ema_tag}_sb{blend_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v19":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        score_decay_tag = int(round(args.score_ema_decay * 100))
        score_start_tag = (
            quant_start_tag
            if args.score_ema_start_period < 0
            else str(args.score_ema_start_period)
        )
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_sd{score_decay_tag}_ss{score_start_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v20":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        ema_tag = int(round(args.stability_ema_decay * 100))
        blend_tag = int(round(args.stability_blend * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_ed{ema_tag}_sb{blend_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v21":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        sample_tag = max(1, int(args.score_collection_interval))
        score_start_tag = (
            quant_start_tag
            if args.score_collection_start_period < 0
            else str(args.score_collection_start_period)
        )
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_si{sample_tag}_ss{score_start_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v22":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        pool_tag = int(round(args.boundary_pool_factor * 10))
        mix_tag = int(round(args.boundary_mix_weight * 100))
        power_tag = int(round(args.boundary_margin_power * 10))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_bp{pool_tag}_bm{mix_tag}_mp{power_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v23":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        sample_tag = max(1, int(args.score_collection_interval))
        score_start_tag = (
            quant_start_tag
            if args.score_collection_start_period < 0
            else str(args.score_collection_start_period)
        )
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_si{sample_tag}_ss{score_start_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v24":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        sample_tag = max(1, int(args.score_collection_interval))
        score_start_tag = (
            quant_start_tag
            if args.score_collection_start_period < 0
            else str(args.score_collection_start_period)
        )
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_si{sample_tag}_ss{score_start_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v25":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        rescue_tag = int(round(args.rescue_fraction * 100))
        phase_specs = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_specs.append(weights)
        if not phase_specs:
            phase_specs = [[float(w) for w in args.committee_weights.split(",") if w.strip()]]
        start_committee_tag = int(round(phase_specs[0][0] * 100))
        end_committee_tag = int(round(phase_specs[-1][0] * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_cm{start_committee_tag}-{end_committee_tag}"
            f"_rf{rescue_tag}_pf{int(round(args.rescue_pool_factor * 10))}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v26":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        phase_specs = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_specs.append(weights)
        if not phase_specs:
            phase_specs = [[float(w) for w in args.committee_weights.split(",") if w.strip()]]
        start_committee_tag = int(round(phase_specs[0][0] * 100))
        end_committee_tag = int(round(phase_specs[-1][0] * 100))
        ema_tag = int(round(args.final_eval_ema_decay * 1000))
        start_epoch_tag = "p" if args.final_eval_start_epoch < 0 else str(args.final_eval_start_epoch)
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_cm{start_committee_tag}-{end_committee_tag}_ed{ema_tag}_se{start_epoch_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v27":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        power_tag = int(round(args.quant_sensitivity_power * 10))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_pw{power_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v28":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        gamma_tag = int(round(args.gamma_modulation_strength * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_gm{gamma_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v29":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        gamma_tag = int(round(args.gamma_modulation_strength * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_gm{gamma_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v30":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        lr_tag = int(round(args.recovery_lr_boost * 100))
        quant_boost_tag = int(round(args.recovery_quant_boost * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}"
            f"_rb{lr_tag}_qb{quant_boost_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v31":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        focus_prune_tag = int(round(args.max_focus_prune_fraction * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_ff{focus_prune_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v32":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        gamma_tag = int(round(args.gamma_modulation_strength * 100))
        d_quant_tag = int(round(args.d_quant_focus_blend * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_gm{gamma_tag}_dq{d_quant_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v33":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        handoff_blend_tag = int(round(args.handoff_anchor_blend * 100))
        handoff_epoch_tag = int(args.handoff_anchor_epochs)
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_hb{handoff_blend_tag}_he{handoff_epoch_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    if args.optimizer == "xai_v34":
        switch_tag = "auto" if args.xai_switch_period < 0 else str(args.xai_switch_period)
        quant_tag = int(round(args.quant_sensitivity_weight * 100))
        quant_start_tag = (
            switch_tag
            if args.quant_sensitivity_start_period < 0
            else str(args.quant_sensitivity_start_period)
        )
        focus_tag = int(round(args.quant_focus_quantile * 100))
        drift_tag = int(round(args.tail_proxy_drift_weight * 100))
        return (
            f"{args.optimizer}_{args.method}_w{args.weight}_sw{switch_tag}"
            f"_qs{quant_tag}_qp{quant_start_tag}_fq{focus_tag}_td{drift_tag}"
            f"_s{args.sparsity}_e{args.epochs}_n{args.train_samples}"
        )
    return (
        f"{args.optimizer}_{args.method}_w{args.weight}_s{args.sparsity}"
        f"_e{args.epochs}_n{args.train_samples}"
    )


def create_quantized_subset_model(model_name: str, device: str):
    if model_name == "vgg7":
        return create_quantized_vgg7(device)
    if model_name == "resnet20":
        return create_quantized_resnet20(device)
    if model_name == "resnet18":
        return create_quantized_resnet18(device)
    raise ValueError(f"Unsupported model: {model_name}")


def create_subset_loaders(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_full = CIFAR10(
        root=args.dataset_root,
        train=True,
        download=False,
        transform=train_transform,
    )
    test_full = CIFAR10(
        root=args.dataset_root,
        train=False,
        download=False,
        transform=test_transform,
    )

    train_idx = torch.randperm(
        len(train_full),
        generator=torch.Generator().manual_seed(args.seed),
    )[: args.train_samples].tolist()
    eval_idx = torch.randperm(
        len(test_full),
        generator=torch.Generator().manual_seed(args.seed + 1),
    )[: args.eval_samples].tolist()

    train_subset = Subset(train_full, train_idx)
    eval_subset = Subset(test_full, eval_idx)

    loader_gen = torch.Generator().manual_seed(args.seed + 10)
    trainloader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=loader_gen,
    )
    testloader = DataLoader(
        eval_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_subset, trainloader, testloader


def build_optimizer(args, oto, model, steps_per_epoch):
    projection_steps = args.projection_epochs * steps_per_epoch
    pruning_steps = args.pruning_epochs * steps_per_epoch
    shared = dict(
        params=oto._graph.get_param_groups(),
        variant="adam",
        lr=args.lr,
        lr_quant=args.lr_quant,
        first_momentum=0.9,
        weight_decay=args.weight_decay,
        target_group_sparsity=args.sparsity,
        start_projection_step=0,
        projection_periods=args.projection_periods,
        projection_steps=projection_steps,
        start_pruning_step=projection_steps,
        pruning_periods=args.pruning_periods,
        pruning_steps=pruning_steps,
        bit_reduction=args.bit_reduction,
        min_bit_wt=args.min_bit,
        max_bit_wt=args.max_bit,
        device=args.device,
    )

    if args.optimizer == "geta":
        return GETA(**shared)

    if args.optimizer == "xai":
        from only_train_once.xai_optimizer import XAI_GETA

        return XAI_GETA(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            **shared,
        )

    if args.optimizer == "xai_v2":
        from only_train_once.xai_optimizer_2 import XAI_GETA_V2

        return XAI_GETA_V2(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            protection_quantile=args.protection_quantile,
            progressive_power=args.progressive_power,
            max_protection_boost=args.max_protection_boost,
            **shared,
        )

    if args.optimizer == "xai_v3":
        from only_train_once.xai_optimizer_3 import XAI_GETA_V3

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        committee_weights = [float(w) for w in args.committee_weights.split(",") if w.strip()]
        return XAI_GETA_V3(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=committee_weights,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            protection_quantile=args.protection_quantile,
            progressive_power=args.progressive_power,
            max_protection_boost=args.max_protection_boost,
            **shared,
        )

    if args.optimizer == "xai_v4":
        from only_train_once.xai_optimizer_4 import XAI_GETA_V4

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        phase_method_weights = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_method_weights.append(weights)
        phase_attr_weights = [
            float(w) for w in args.phase_attribution_weights.split(",") if w.strip()
        ]
        return XAI_GETA_V4(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=[float(w) for w in args.committee_weights.split(",") if w.strip()],
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            protection_quantile=args.protection_quantile,
            progressive_power=args.progressive_power,
            max_protection_boost=args.max_protection_boost,
            phase_method_weights=phase_method_weights,
            phase_attribution_weights=phase_attr_weights,
            **shared,
        )

    if args.optimizer == "xai_v5":
        from only_train_once.xai_optimizer_5 import XAI_GETA_V5

        return XAI_GETA_V5(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            xai_decay_periods=args.xai_decay_periods,
            **shared,
        )

    if args.optimizer == "xai_v6":
        from only_train_once.xai_optimizer_6 import XAI_GETA_V6

        return XAI_GETA_V6(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            protection_quantile=args.protection_quantile,
            progressive_power=args.progressive_power,
            max_protection_boost=args.max_protection_boost,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            xai_decay_periods=args.xai_decay_periods,
            **shared,
        )

    if args.optimizer == "xai_v7":
        from only_train_once.xai_optimizer_7 import XAI_GETA_V7

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        committee_weights = [float(w) for w in args.committee_weights.split(",") if w.strip()]
        return XAI_GETA_V7(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=committee_weights,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            protection_quantile=args.protection_quantile,
            progressive_power=args.progressive_power,
            max_protection_boost=args.max_protection_boost,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            xai_decay_periods=args.xai_decay_periods,
            **shared,
        )

    if args.optimizer == "xai_v8":
        from only_train_once.xai_optimizer_8 import XAI_GETA_V8

        return XAI_GETA_V8(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            xai_decay_periods=args.xai_decay_periods,
            modulation_clip_start=args.modulation_clip_start,
            modulation_clip_end=args.modulation_clip_end,
            **shared,
        )

    if args.optimizer == "xai_v9":
        from only_train_once.xai_optimizer_9 import XAI_GETA_V9

        return XAI_GETA_V9(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            protection_quantile=args.protection_quantile,
            progressive_power=args.progressive_power,
            max_protection_boost=args.max_protection_boost,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            late_protection_scale=args.late_protection_scale,
            **shared,
        )

    if args.optimizer == "xai_v10":
        from only_train_once.xai_optimizer_10 import XAI_GETA_V10

        return XAI_GETA_V10(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            rescue_quantile=args.protection_quantile,
            rescue_fraction=args.rescue_fraction,
            rescue_pool_factor=args.rescue_pool_factor,
            **shared,
        )

    if args.optimizer == "xai_v11":
        from only_train_once.xai_optimizer_11 import XAI_GETA_V11

        return XAI_GETA_V11(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            **shared,
        )

    if args.optimizer == "xai_v12":
        from only_train_once.xai_optimizer_12 import XAI_GETA_V12

        return XAI_GETA_V12(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            **shared,
        )

    if args.optimizer == "xai_v13":
        from only_train_once.xai_optimizer_13 import XAI_GETA_V13

        return XAI_GETA_V13(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            stability_ema_decay=args.stability_ema_decay,
            stability_blend=args.stability_blend,
            **shared,
        )

    if args.optimizer == "xai_v14":
        from only_train_once.xai_optimizer_14 import XAI_GETA_V14

        return XAI_GETA_V14(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            **shared,
        )

    if args.optimizer == "xai_v15":
        from only_train_once.xai_optimizer_15 import XAI_GETA_V15

        return XAI_GETA_V15(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            rescue_fraction=args.rescue_fraction,
            rescue_pool_factor=args.rescue_pool_factor,
            **shared,
        )

    if args.optimizer == "xai_v16":
        from only_train_once.xai_optimizer_16 import XAI_GETA_V16

        return XAI_GETA_V16(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            quant_focus_start_quantile=(
                None
                if args.quant_focus_start_quantile < 0
                else args.quant_focus_start_quantile
            ),
            **shared,
        )

    if args.optimizer == "xai_v17":
        from only_train_once.xai_optimizer_17 import XAI_GETA_V17

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        committee_weights = [float(w) for w in args.committee_weights.split(",") if w.strip()]
        phase_method_weights = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_method_weights.append(weights)
        return XAI_GETA_V17(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=committee_weights,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            phase_method_weights=phase_method_weights or None,
            **shared,
        )

    if args.optimizer == "xai_v18":
        from only_train_once.xai_optimizer_18 import XAI_GETA_V18

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        committee_weights = [float(w) for w in args.committee_weights.split(",") if w.strip()]
        phase_method_weights = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_method_weights.append(weights)
        return XAI_GETA_V18(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=committee_weights,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            phase_method_weights=phase_method_weights or None,
            stability_ema_decay=args.stability_ema_decay,
            stability_blend=args.stability_blend,
            **shared,
        )

    if args.optimizer == "xai_v19":
        from only_train_once.xai_optimizer_19 import XAI_GETA_V19

        return XAI_GETA_V19(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            score_ema_decay=args.score_ema_decay,
            score_ema_start_period=(
                None
                if args.score_ema_start_period < 0
                else args.score_ema_start_period
            ),
            **shared,
        )

    if args.optimizer == "xai_v20":
        from only_train_once.xai_optimizer_20 import XAI_GETA_V20

        return XAI_GETA_V20(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            stability_ema_decay=args.stability_ema_decay,
            stability_blend=args.stability_blend,
            **shared,
        )

    if args.optimizer == "xai_v21":
        from only_train_once.xai_optimizer_21 import XAI_GETA_V21

        return XAI_GETA_V21(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            score_collection_interval=args.score_collection_interval,
            score_collection_start_period=(
                None
                if args.score_collection_start_period < 0
                else args.score_collection_start_period
            ),
            **shared,
        )

    if args.optimizer == "xai_v22":
        from only_train_once.xai_optimizer_22 import XAI_GETA_V22

        return XAI_GETA_V22(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            boundary_pool_factor=args.boundary_pool_factor,
            boundary_mix_weight=args.boundary_mix_weight,
            boundary_margin_power=args.boundary_margin_power,
            **shared,
        )

    if args.optimizer == "xai_v23":
        from only_train_once.xai_optimizer_23 import XAI_GETA_V23

        return XAI_GETA_V23(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            score_collection_interval=args.score_collection_interval,
            score_collection_start_period=(
                None
                if args.score_collection_start_period < 0
                else args.score_collection_start_period
            ),
            **shared,
        )

    if args.optimizer == "xai_v24":
        from only_train_once.xai_optimizer_24 import XAI_GETA_V24

        return XAI_GETA_V24(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            score_collection_interval=args.score_collection_interval,
            score_collection_start_period=(
                None
                if args.score_collection_start_period < 0
                else args.score_collection_start_period
            ),
            **shared,
        )

    if args.optimizer == "xai_v25":
        from only_train_once.xai_optimizer_25 import XAI_GETA_V25

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        committee_weights = [float(w) for w in args.committee_weights.split(",") if w.strip()]
        phase_method_weights = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_method_weights.append(weights)
        return XAI_GETA_V25(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=committee_weights,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            rescue_fraction=args.rescue_fraction,
            rescue_pool_factor=args.rescue_pool_factor,
            phase_method_weights=phase_method_weights or None,
            **shared,
        )

    if args.optimizer == "xai_v26":
        from only_train_once.xai_optimizer_26 import XAI_GETA_V26

        committee_methods = [m.strip() for m in args.committee_methods.split(",") if m.strip()]
        committee_weights = [float(w) for w in args.committee_weights.split(",") if w.strip()]
        phase_method_weights = []
        for phase_weights in args.phase_method_weights.split(";"):
            weights = [float(w) for w in phase_weights.split(",") if w.strip()]
            if weights:
                phase_method_weights.append(weights)
        final_eval_start_step = None
        if args.final_eval_start_epoch > 0:
            final_eval_start_step = max(0, (args.final_eval_start_epoch - 1) * steps_per_epoch + 1)
        return XAI_GETA_V26(
            model=model,
            attribution_method=committee_methods[0],
            attribution_methods=committee_methods,
            attribution_method_weights=committee_weights,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            final_eval_ema_decay=args.final_eval_ema_decay,
            final_eval_start_step=final_eval_start_step,
            phase_method_weights=phase_method_weights or None,
            **shared,
        )

    if args.optimizer == "xai_v27":
        from only_train_once.xai_optimizer_27 import XAI_GETA_V27

        return XAI_GETA_V27(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            quant_sensitivity_power=args.quant_sensitivity_power,
            **shared,
        )

    if args.optimizer == "xai_v28":
        from only_train_once.xai_optimizer_28 import XAI_GETA_V28

        return XAI_GETA_V28(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            gamma_modulation_strength=args.gamma_modulation_strength,
            **shared,
        )

    if args.optimizer == "xai_v29":
        from only_train_once.xai_optimizer_29 import XAI_GETA_V29

        return XAI_GETA_V29(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            gamma_modulation_strength=args.gamma_modulation_strength,
            **shared,
        )

    if args.optimizer == "xai_v30":
        from only_train_once.xai_optimizer_30 import XAI_GETA_V30

        return XAI_GETA_V30(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            total_training_steps=args.epochs * steps_per_epoch,
            recovery_lr_boost=args.recovery_lr_boost,
            recovery_quant_boost=args.recovery_quant_boost,
            **shared,
        )

    if args.optimizer == "xai_v31":
        from only_train_once.xai_optimizer_31 import XAI_GETA_V31

        return XAI_GETA_V31(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            max_focus_prune_fraction=args.max_focus_prune_fraction,
            **shared,
        )

    if args.optimizer == "xai_v32":
        from only_train_once.xai_optimizer_32 import XAI_GETA_V32

        return XAI_GETA_V32(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            gamma_modulation_strength=args.gamma_modulation_strength,
            d_quant_focus_blend=args.d_quant_focus_blend,
            **shared,
        )

    if args.optimizer == "xai_v33":
        from only_train_once.xai_optimizer_33 import XAI_GETA_V33

        return XAI_GETA_V33(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            handoff_anchor_blend=args.handoff_anchor_blend,
            handoff_anchor_steps=max(0, args.handoff_anchor_epochs * steps_per_epoch),
            **shared,
        )

    if args.optimizer == "xai_v34":
        from only_train_once.xai_optimizer_34 import XAI_GETA_V34

        return XAI_GETA_V34(
            model=model,
            attribution_method=args.method,
            attribution_weight=args.weight,
            ema_decay=args.ema_decay,
            compute_attribution_freq=args.attribution_freq,
            attribution_n_steps=args.attribution_n_steps,
            xai_switch_period=(
                None if args.xai_switch_period < 0 else args.xai_switch_period
            ),
            quant_sensitivity_weight=args.quant_sensitivity_weight,
            quant_sensitivity_start_period=(
                None
                if args.quant_sensitivity_start_period < 0
                else args.quant_sensitivity_start_period
            ),
            quant_focus_quantile=args.quant_focus_quantile,
            tail_proxy_drift_weight=args.tail_proxy_drift_weight,
            **shared,
        )

    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def maybe_compute_initial_attributions(args, optimizer, train_subset):
    if args.optimizer == "geta":
        return
    small_loader = DataLoader(
        train_subset,
        batch_size=16,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed + 99),
    )
    optimizer.compute_initial_attributions(small_loader, num_batches=2)


def write_curve_csv(path: str, rows: List[EpochMetric]):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_scheduler(args, optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma,
        )
    if args.scheduler == "multistep":
        milestones = [
            int(item) for item in args.scheduler_milestones.split(",") if item.strip()
        ]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.scheduler_gamma,
        )
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.scheduler_min_lr,
        )
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def main():
    args = parse_args()
    seed_all(args.seed)
    run_name = build_run_name(args)
    if args.model != "vgg7" and not args.run_name:
        run_name = f"{args.model}_{run_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    train_subset, trainloader, testloader = create_subset_loaders(args)
    model, dummy_input = create_quantized_subset_model(args.model, args.device)
    oto = OTO(model, dummy_input)
    optimizer = build_optimizer(args, oto, model, len(trainloader))
    oto._optimizer = optimizer
    maybe_compute_initial_attributions(args, optimizer, train_subset)

    criterion = nn.CrossEntropyLoss()
    scheduler = build_scheduler(args, optimizer)
    log_epochs = {int(item) for item in args.log_epochs.split(",") if item.strip()}

    history: List[EpochMetric] = []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in trainloader:
            X = X.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            if args.optimizer == "geta":
                optimizer.step()
            else:
                optimizer.step(inputs=X, targets=y)
            epoch_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        acc1, acc5 = evaluate_model(model, testloader, args.device)
        metrics = optimizer.compute_metrics()
        row = EpochMetric(
            epoch=epoch,
            loss=round(epoch_loss / len(trainloader), 4),
            acc1=round(acc1, 2),
            acc5=round(acc5, 2),
            group_sparsity=round(metrics.group_sparsity, 4),
            num_important_groups=int(metrics.num_important_groups),
            num_redundant_groups=int(metrics.num_redundant_groups),
        )
        history.append(row)
        if hasattr(optimizer, "observe_epoch"):
            optimizer.observe_epoch(
                epoch=epoch,
                train_loss=row.loss,
                group_sparsity=row.group_sparsity,
            )
        if epoch in log_epochs:
            print(
                f"{run_name:30s} ep{epoch:02d}: loss={row.loss:.4f} "
                f"acc1={row.acc1:.2f} acc5={row.acc5:.2f} "
                f"sparsity={row.group_sparsity:.4f}"
            )

    if hasattr(optimizer, "finalize_for_evaluation"):
        finalized = optimizer.finalize_for_evaluation()
        if finalized:
            final_acc1, final_acc5 = evaluate_model(model, testloader, args.device)
            history[-1].acc1 = round(final_acc1, 2)
            history[-1].acc5 = round(final_acc5, 2)

    best_epoch = max(history, key=lambda item: item.acc1)
    summary = {
        "run_name": run_name,
        "model_name": args.model,
        "optimizer": args.optimizer,
        "method": args.method,
        "weight": args.weight,
        "committee_methods": args.committee_methods,
        "committee_weights": args.committee_weights,
        "train_samples": args.train_samples,
        "eval_samples": args.eval_samples,
        "epochs": args.epochs,
        "scheduler": args.scheduler,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "scheduler_milestones": args.scheduler_milestones,
        "scheduler_min_lr": args.scheduler_min_lr,
        "projection_epochs": args.projection_epochs,
        "pruning_epochs": args.pruning_epochs,
        "target_sparsity": args.sparsity,
        "best_acc1": best_epoch.acc1,
        "best_epoch": best_epoch.epoch,
        "final_acc1": history[-1].acc1,
        "final_acc5": history[-1].acc5,
        "final_sparsity": history[-1].group_sparsity,
        "runtime_s": round(time.time() - start_time, 1),
    }

    curve_csv = os.path.join(args.output_dir, f"{run_name}_curve.csv")
    summary_json = os.path.join(args.output_dir, f"{run_name}_summary.json")
    write_curve_csv(curve_csv, history)
    with open(summary_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("JSON_RESULTS=" + json.dumps(summary))
    print(f"curve_csv={curve_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
