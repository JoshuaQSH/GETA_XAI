import argparse
import glob
import json

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot cross-model GETA vs XAI smoke summaries."
    )
    parser.add_argument("--summary-glob", required=True, help="Glob for summary JSON files.")
    parser.add_argument("--acc-output", required=True, help="Output PNG for grouped acc@1 bars.")
    parser.add_argument("--gain-output", required=True, help="Output PNG for final gain bars.")
    parser.add_argument(
        "--title-prefix",
        default="Cross-model CIFAR-10 smoke",
        help="Prefix for plot titles.",
    )
    return parser.parse_args()


def load_rows(summary_glob):
    rows = []
    for path in sorted(glob.glob(summary_glob)):
        with open(path) as fh:
            row = json.load(fh)
        if row.get("optimizer") not in {"geta", "xai_v14"}:
            continue
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No matching summary JSON files for {summary_glob}")
    return rows


def plot_acc_summary(rows, output_path, title_prefix):
    models = sorted({row.get("model_name", "vgg7") for row in rows})
    x = list(range(len(models)))
    width = 0.34
    geta_final = []
    xai_final = []
    geta_best = []
    xai_best = []
    for model in models:
        per_model = {
            row["optimizer"]: row for row in rows if row.get("model_name", "vgg7") == model
        }
        geta = per_model["geta"]
        xai = per_model["xai_v14"]
        geta_final.append(geta["final_acc1"])
        xai_final.append(xai["final_acc1"])
        geta_best.append(geta["best_acc1"])
        xai_best.append(xai["best_acc1"])

    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width / 2 for i in x], geta_final, width=width, label="GETA final", color="#4C78A8")
    plt.bar([i + width / 2 for i in x], xai_final, width=width, label="XAI-V14 final", color="#F58518")
    plt.scatter([i - width / 2 for i in x], geta_best, color="#1f4e79", marker="o", zorder=3, label="GETA best")
    plt.scatter([i + width / 2 for i in x], xai_best, color="#b85c00", marker="o", zorder=3, label="XAI-V14 best")
    plt.xticks(x, [model.upper() for model in models])
    plt.ylabel("acc@1")
    plt.title(f"{title_prefix}: GETA vs XAI-V14")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_gain_summary(rows, output_path, title_prefix):
    models = sorted({row.get("model_name", "vgg7") for row in rows})
    gains = []
    for model in models:
        per_model = {
            row["optimizer"]: row for row in rows if row.get("model_name", "vgg7") == model
        }
        gains.append(per_model["xai_v14"]["final_acc1"] - per_model["geta"]["final_acc1"])

    x = list(range(len(models)))
    colors = ["#54A24B" if gain >= 0 else "#E45756" for gain in gains]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(x, gains, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, [model.upper() for model in models])
    plt.ylabel("final acc@1 gain over GETA")
    plt.title(f"{title_prefix}: XAI-V14 final gain")
    for bar, gain in zip(bars, gains):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            gain + (0.15 if gain >= 0 else -0.4),
            f"{gain:+.1f}",
            ha="center",
            va="bottom" if gain >= 0 else "top",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    rows = load_rows(args.summary_glob)
    plot_acc_summary(rows, args.acc_output, args.title_prefix)
    plot_gain_summary(rows, args.gain_output, args.title_prefix)
    print(f"Saved acc summary to {args.acc_output}")
    print(f"Saved gain summary to {args.gain_output}")


if __name__ == "__main__":
    main()
