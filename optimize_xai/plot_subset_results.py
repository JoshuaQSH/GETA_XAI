import argparse
import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot subset experiment summaries and loss curves."
    )
    parser.add_argument(
        "--curve-glob",
        required=True,
        help="Glob for curve CSV files, e.g. outputs/update_optimize/*_curve.csv",
    )
    parser.add_argument(
        "--summary-glob",
        required=True,
        help="Glob for summary JSON files, e.g. outputs/update_optimize/*_summary.json",
    )
    parser.add_argument(
        "--acc-output",
        required=True,
        help="Path to the output acc@1 summary PNG.",
    )
    parser.add_argument(
        "--loss-output",
        required=True,
        help="Path to the output loss curve PNG.",
    )
    parser.add_argument(
        "--acc-title",
        default="VGG7 subset best/final acc@1 by variant",
        help="Title for the acc@1 summary plot.",
    )
    parser.add_argument(
        "--loss-title",
        default="VGG7 subset loss over epochs",
        help="Title for the loss curve plot.",
    )
    parser.add_argument(
        "--only-run-names",
        default="",
        help="Comma-separated run names to include. Empty means include everything matched by the globs.",
    )
    return parser.parse_args()


def _method_tag(method):
    mapping = {
        "saliency": "Sal",
        "deep_lift": "DL",
        "input_x_gradient": "IxG",
        "layer_gradient_x_activation": "GxA",
        "layer_conductance": "LC",
        "layer_integrated_gradients": "LIG",
    }
    return mapping.get(method, method[:4].title())


def _short_label(summary):
    optimizer = summary["optimizer"]
    method = summary.get("method", "na")
    weight = summary.get("weight", 0.0)
    run_name = summary.get("run_name", "")
    if optimizer == "geta":
        return "GETA"
    if optimizer == "xai":
        return f"{_method_tag(method)}{weight:.1f}"
    if optimizer == "xai_v2":
        return f"V2{weight:.1f}"
    if optimizer == "xai_v3":
        weights = summary.get("committee_weights", "")
        try:
            sal_weight = int(round(float(weights.split(",")[0]) * 100))
        except (IndexError, ValueError):
            sal_weight = 0
        return f"V3C{sal_weight}"
    if optimizer == "xai_v5":
        return f"V5{weight:.1f}"
    if optimizer == "xai_v6":
        return f"V6{weight:.1f}"
    if optimizer == "xai_v7":
        weights = summary.get("committee_weights", "")
        try:
            sal_weight = int(round(float(weights.split(",")[0]) * 100))
        except (IndexError, ValueError):
            sal_weight = 0
        return f"V7C{sal_weight}"
    if optimizer == "xai_v8":
        switch = re.search(r"_sw(\d+)", run_name)
        clip = re.search(r"_cs(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        clip_tag = clip.group(1) if clip else "x"
        return f"V8s{switch_tag}c{clip_tag}"
    if optimizer == "xai_v9":
        switch = re.search(r"_sw(\d+)", run_name)
        late = re.search(r"_lp(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        late_tag = late.group(1) if late else "x"
        return f"V9s{switch_tag}p{late_tag}"
    if optimizer == "xai_v10":
        switch = re.search(r"_sw(\d+)", run_name)
        rescue = re.search(r"_rf(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        rescue_tag = rescue.group(1) if rescue else "x"
        return f"V10s{switch_tag}r{rescue_tag}"
    if optimizer == "xai_v11":
        switch = re.search(r"_sw(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        return f"V11s{switch_tag}"
    if optimizer == "xai_v12":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        return f"V12s{switch_tag}q{quant_tag}"
    if optimizer == "xai_v13":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        blend = re.search(r"_sb(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        blend_tag = blend.group(1) if blend else "x"
        return f"V13s{switch_tag}q{quant_tag}b{blend_tag}"
    if optimizer == "xai_v14":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        return f"V14s{switch_tag}q{quant_tag}f{focus_tag}"
    if optimizer == "xai_v15":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        rescue = re.search(r"_rf(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        rescue_tag = rescue.group(1) if rescue else "x"
        return f"V15s{switch_tag}q{quant_tag}f{focus_tag}r{rescue_tag}"
    if optimizer == "xai_v16":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus_start = re.search(r"_fs(\d+)", run_name)
        focus_end = re.search(r"_fe(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_start_tag = focus_start.group(1) if focus_start else "x"
        focus_end_tag = focus_end.group(1) if focus_end else "x"
        return f"V16s{switch_tag}q{quant_tag}f{focus_start_tag}-{focus_end_tag}"
    if optimizer == "xai_v17":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        committee = re.search(r"_cm(\d+)-(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        committee_tag = (
            f"{committee.group(1)}-{committee.group(2)}" if committee else "x"
        )
        return f"V17s{switch_tag}q{quant_tag}f{focus_tag}c{committee_tag}"
    if optimizer == "xai_v18":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        committee = re.search(r"_cm(\d+)-(\d+)", run_name)
        blend = re.search(r"_sb(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        committee_tag = (
            f"{committee.group(1)}-{committee.group(2)}" if committee else "x"
        )
        blend_tag = blend.group(1) if blend else "x"
        return f"V18s{switch_tag}q{quant_tag}f{focus_tag}c{committee_tag}b{blend_tag}"
    if optimizer == "xai_v19":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        score_decay = re.search(r"_sd(\d+)", run_name)
        score_start = re.search(r"_ss(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        score_decay_tag = score_decay.group(1) if score_decay else "x"
        score_start_tag = score_start.group(1) if score_start else "x"
        return f"V19s{switch_tag}q{quant_tag}f{focus_tag}d{score_decay_tag}p{score_start_tag}"
    if optimizer == "xai_v20":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        blend = re.search(r"_sb(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        blend_tag = blend.group(1) if blend else "x"
        return f"V20s{switch_tag}q{quant_tag}f{focus_tag}b{blend_tag}"
    if optimizer == "xai_v21":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        sample = re.search(r"_si(\d+)", run_name)
        start = re.search(r"_ss(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        sample_tag = sample.group(1) if sample else "x"
        start_tag = start.group(1) if start else "x"
        return f"V21s{switch_tag}q{quant_tag}f{focus_tag}i{sample_tag}p{start_tag}"
    if optimizer == "xai_v22":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        pool = re.search(r"_bp(\d+)", run_name)
        mix = re.search(r"_bm(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        pool_tag = pool.group(1) if pool else "x"
        mix_tag = mix.group(1) if mix else "x"
        return f"V22s{switch_tag}q{quant_tag}f{focus_tag}p{pool_tag}m{mix_tag}"
    if optimizer == "xai_v23":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        sample = re.search(r"_si(\d+)", run_name)
        start = re.search(r"_ss(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        sample_tag = sample.group(1) if sample else "x"
        start_tag = start.group(1) if start else "x"
        return f"V23s{switch_tag}q{quant_tag}f{focus_tag}i{sample_tag}p{start_tag}"
    if optimizer == "xai_v24":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        sample = re.search(r"_si(\d+)", run_name)
        start = re.search(r"_ss(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        sample_tag = sample.group(1) if sample else "x"
        start_tag = start.group(1) if start else "x"
        return f"V24s{switch_tag}q{quant_tag}f{focus_tag}i{sample_tag}p{start_tag}"
    if optimizer == "xai_v25":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        committee = re.search(r"_cm(\d+)-(\d+)", run_name)
        rescue = re.search(r"_rf(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        committee_tag = (
            f"{committee.group(1)}-{committee.group(2)}" if committee else "x"
        )
        rescue_tag = rescue.group(1) if rescue else "x"
        return f"V25s{switch_tag}q{quant_tag}f{focus_tag}c{committee_tag}r{rescue_tag}"
    if optimizer == "xai_v26":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        committee = re.search(r"_cm(\d+)-(\d+)", run_name)
        ema = re.search(r"_ed(\d+)", run_name)
        start = re.search(r"_se([a-z0-9]+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        committee_tag = (
            f"{committee.group(1)}-{committee.group(2)}" if committee else "x"
        )
        ema_tag = ema.group(1) if ema else "x"
        start_tag = start.group(1) if start else "x"
        return f"V26s{switch_tag}q{quant_tag}f{focus_tag}c{committee_tag}e{ema_tag}t{start_tag}"
    if optimizer == "xai_v27":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        power = re.search(r"_pw(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        power_tag = power.group(1) if power else "x"
        return f"V27s{switch_tag}q{quant_tag}f{focus_tag}p{power_tag}"
    if optimizer == "xai_v28":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        gamma = re.search(r"_gm(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        gamma_tag = gamma.group(1) if gamma else "x"
        return f"V28s{switch_tag}q{quant_tag}f{focus_tag}g{gamma_tag}"
    if optimizer == "xai_v29":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        gamma = re.search(r"_gm(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        gamma_tag = gamma.group(1) if gamma else "x"
        return f"V29s{switch_tag}q{quant_tag}f{focus_tag}g{gamma_tag}"
    if optimizer == "xai_v30":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        lr_boost = re.search(r"_rb(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        lr_tag = lr_boost.group(1) if lr_boost else "x"
        return f"V30s{switch_tag}q{quant_tag}f{focus_tag}r{lr_tag}"
    if optimizer == "xai_v31":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        prune_cap = re.search(r"_ff(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        prune_cap_tag = prune_cap.group(1) if prune_cap else "x"
        return f"V31s{switch_tag}q{quant_tag}f{focus_tag}p{prune_cap_tag}"
    if optimizer == "xai_v32":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        gamma = re.search(r"_gm(\d+)", run_name)
        d_quant = re.search(r"_dq(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        gamma_tag = gamma.group(1) if gamma else "x"
        d_quant_tag = d_quant.group(1) if d_quant else "x"
        return f"V32s{switch_tag}q{quant_tag}f{focus_tag}g{gamma_tag}d{d_quant_tag}"
    if optimizer == "xai_v33":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        handoff_blend = re.search(r"_hb(\d+)", run_name)
        handoff_epochs = re.search(r"_he(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        handoff_blend_tag = handoff_blend.group(1) if handoff_blend else "x"
        handoff_epoch_tag = handoff_epochs.group(1) if handoff_epochs else "x"
        return f"V33s{switch_tag}q{quant_tag}f{focus_tag}h{handoff_blend_tag}e{handoff_epoch_tag}"
    if optimizer == "xai_v34":
        switch = re.search(r"_sw(\d+)", run_name)
        quant = re.search(r"_qs(\d+)", run_name)
        focus = re.search(r"_fq(\d+)", run_name)
        drift = re.search(r"_td(\d+)", run_name)
        switch_tag = switch.group(1) if switch else "x"
        quant_tag = quant.group(1) if quant else "x"
        focus_tag = focus.group(1) if focus else "x"
        drift_tag = drift.group(1) if drift else "x"
        return f"V34s{switch_tag}q{quant_tag}f{focus_tag}d{drift_tag}"
    return os.path.basename(summary["run_name"])[:12]


def _load_summaries(summary_glob):
    paths = sorted(glob.glob(summary_glob))
    if not paths:
        raise FileNotFoundError(f"No summary JSON files matched: {summary_glob}")
    rows = []
    for path in paths:
        with open(path) as fh:
            summary = json.load(fh)
        summary["short_label"] = _short_label(summary)
        rows.append(summary)
    rows.sort(key=lambda item: (item["optimizer"], item["method"], item["weight"], item["run_name"]))
    return rows


def _load_curves(curve_glob):
    paths = sorted(glob.glob(curve_glob))
    if not paths:
        raise FileNotFoundError(f"No curve CSV files matched: {curve_glob}")
    curves = []
    for path in paths:
        run_name = os.path.basename(path).replace("_curve.csv", "")
        epochs = []
        losses = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                epochs.append(int(row["epoch"]))
                losses.append(float(row["loss"]))
        curves.append(
            {
                "run_name": run_name,
                "epochs": epochs,
                "losses": losses,
            }
        )
    return curves


def _filter_by_run_names(items, only_run_names):
    if not only_run_names:
        return items
    allowed = {name.strip() for name in only_run_names.split(",") if name.strip()}
    return [item for item in items if item["run_name"] in allowed]


def _select_reference_geta(summaries):
    geta_rows = [row for row in summaries if row["optimizer"] == "geta"]
    if not geta_rows:
        raise ValueError("No GETA baseline runs found.")
    return max(
        geta_rows,
        key=lambda item: (item["final_acc1"], item["best_acc1"], item["epochs"]),
    )


def _select_variants(summaries):
    variants = []
    for row in summaries:
        if row["optimizer"] == "geta":
            continue
        row = dict(row)
        variants.append(row)

    variants.sort(
        key=lambda item: (
            item["final_acc1"],
            item["best_acc1"],
            item["epochs"],
            item["short_label"],
        )
    )
    return variants


def plot_acc_summary(summaries, output_path, title):
    baseline = _select_reference_geta(summaries)
    variants = _select_variants(summaries)
    if not variants:
        raise ValueError("No non-GETA variants found to plot.")

    labels = [row["short_label"] for row in variants]
    x = list(range(len(labels)))
    matched_geta_final = [baseline["final_acc1"] for _ in variants]
    best_acc = [row["best_acc1"] for row in variants]
    final_acc = [row["final_acc1"] for row in variants]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 6))
    plt.plot(
        x,
        matched_geta_final,
        marker="o",
        linestyle="--",
        color="black",
        linewidth=1.8,
        label=f"GETA final ({baseline['short_label']})",
    )
    plt.plot(x, best_acc, marker="o", linewidth=2.2, label="Variant best acc@1")
    plt.plot(x, final_acc, marker="o", linewidth=2.2, label="Variant final acc@1")
    plt.ylabel("Acc@1")
    plt.xlabel("Variant")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved acc summary to {output_path}")


def plot_loss_curves(curves, summaries, output_path, title):
    baseline = _select_reference_geta(summaries)
    variants = _select_variants(summaries)
    selected_runs = {baseline["run_name"], *[row["run_name"] for row in variants]}
    label_by_run = {row["run_name"]: row["short_label"] for row in summaries}
    plt.figure(figsize=(10, 6))
    for curve in curves:
        if curve["run_name"] not in selected_runs:
            continue
        label = label_by_run.get(curve["run_name"], curve["run_name"])
        is_baseline = curve["run_name"] == baseline["run_name"]
        plt.plot(
            curve["epochs"],
            curve["losses"],
            label=label,
            linewidth=2,
            linestyle=":" if is_baseline else "-",
            alpha=1.0 if is_baseline else 0.6,
            color="black" if is_baseline else None,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {output_path}")


def main():
    args = parse_args()
    summaries = _filter_by_run_names(_load_summaries(args.summary_glob), args.only_run_names)
    curves = _filter_by_run_names(_load_curves(args.curve_glob), args.only_run_names)
    plot_acc_summary(summaries, args.acc_output, args.acc_title)
    plot_loss_curves(curves, summaries, args.loss_output, args.loss_title)


if __name__ == "__main__":
    main()
