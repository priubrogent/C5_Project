"""
Plot pretrained vs fine-tuned SAM comparison across all task_a prompt strategies.

Run:
    python plot_task_a_comparison.py
    python plot_task_a_comparison.py --output_dir outputs/plots
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def R(map5095, map50, map75, map_ped, map_car):
    return dict(map5095=map5095, map50=map50, map75=map75,
                map_ped=map_ped, map_car=map_car)

EXPERIMENTS = [
    # (display_label,       group,          pretrained_results,                finetuned_results)
    ("BBox Center",         "Single point", R(0.3090, 0.4864, 0.3371, 0.1018, 0.5161), R(0.3295, 0.5161, 0.3528, 0.1245, 0.5344)),
    ("Mask Centroid",       "Single point", R(0.3245, 0.5130, 0.3498, 0.1082, 0.5407), R(0.3491, 0.5515, 0.3750, 0.1428, 0.5554)),
    ("Rand Mask n=1",       "Random Mask",  R(0.2671, 0.4387, 0.2881, 0.1053, 0.4290), R(0.2986, 0.4928, 0.3180, 0.1477, 0.4495)),
    ("Rand Mask n=3",       "Random Mask",  R(0.4963, 0.7854, 0.5320, 0.3852, 0.6074), R(0.4908, 0.7838, 0.5145, 0.3778, 0.6037)),
    ("Rand Mask n=5",       "Random Mask",  R(0.4693, 0.7693, 0.4781, 0.3707, 0.5679), R(0.4415, 0.7364, 0.4413, 0.3290, 0.5540)),
    ("Rand BBox n=1",       "Random BBox",  R(0.2107, 0.3461, 0.2291, 0.0799, 0.3415), R(0.2281, 0.3729, 0.2438, 0.1045, 0.3517)),
    ("Rand BBox n=3",       "Random BBox",  R(0.3978, 0.6651, 0.4031, 0.2861, 0.5096), R(0.3642, 0.6276, 0.3587, 0.2442, 0.4842)),
    ("Rand BBox n=5",       "Random BBox",  R(0.3424, 0.6047, 0.3304, 0.2272, 0.4575), R(0.2948, 0.5364, 0.2756, 0.1662, 0.4233)),
    # ("SIFT Best",           "SIFT",         R(0.2540, 0.4375, 0.2654, 0.1562, 0.3518), R(0.2825, 0.4872, 0.2927, 0.1974, 0.3676)),
    # ("SIFT TopK n=1",       "SIFT",         R(0.2540, 0.4375, 0.2654, 0.1562, 0.3518), R(0.2825, 0.4872, 0.2927, 0.1974, 0.3676)),
    # ("SIFT TopK n=3",       "SIFT",         None,                                       R(0.4580, 0.7507, 0.4842, 0.3723, 0.5438)),
    # ("SIFT TopK n=5",       "SIFT",         None,                                       R(0.4306, 0.7233, 0.4367, 0.3396, 0.5217)),
    ("GT BBox",             "GT BBox",      R(0.5740, 0.9026, 0.6339, 0.4530, 0.6949), R(0.5843, 0.9037, 0.6488, 0.4770, 0.7255)),
]

METRICS = [
    ("map5095", "mAP @ 0.50:0.95"),
    ("map50",   "mAP @ 0.50"),
    ("map75",   "mAP @ 0.75"),
    ("map_ped", "mAP Pedestrian"),
    ("map_car", "mAP Car"),
]

COLOR_PRETRAINED = "#4C72B0"
COLOR_FINETUNED  = "#DD8452"
COLOR_NONE       = "#CCCCCC"   # bar color when one side is missing


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_grouped_bar(ax, metric_key, metric_label, experiments, bar_width=0.35):
    labels    = [e[0] for e in experiments]
    pre_vals  = [e[2][metric_key] if e[2] is not None else None for e in experiments]
    ft_vals   = [e[3][metric_key] if e[3] is not None else None for e in experiments]

    x = np.arange(len(labels))

    for i, (pre, ft) in enumerate(zip(pre_vals, ft_vals)):
        pre_color = COLOR_PRETRAINED if pre is not None else COLOR_NONE
        ft_color  = COLOR_FINETUNED  if ft  is not None else COLOR_NONE
        pre_val   = pre if pre is not None else 0.0
        ft_val    = ft  if ft  is not None else 0.0

        b1 = ax.bar(x[i] - bar_width / 2, pre_val, bar_width, color=pre_color, zorder=3)
        b2 = ax.bar(x[i] + bar_width / 2, ft_val,  bar_width, color=ft_color,  zorder=3)

        # Value labels on bars
        if pre is not None:
            ax.text(x[i] - bar_width / 2, pre_val + 0.005, f"{pre_val:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
        if ft is not None:
            ax.text(x[i] + bar_width / 2, ft_val + 0.005, f"{ft_val:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)

        # Delta annotation above the pair
        if pre is not None and ft is not None:
            delta = ft_val - pre_val
            sign  = "+" if delta >= 0 else ""
            top   = max(pre_val, ft_val) + 0.06
            color = "green" if delta >= 0 else "red"
            ax.text(x[i], top, f"{sign}{delta:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Group separators
    groups = [e[1] for e in experiments]
    prev_group = None
    for i, g in enumerate(groups):
        if g != prev_group and i > 0:
            ax.axvline(x[i] - 0.5, color="grey", linewidth=0.8, linestyle=":", zorder=2)
        prev_group = g

    # Group labels at top
    from itertools import groupby
    group_spans = []
    idx = 0
    for grp, items in groupby(enumerate(experiments), key=lambda t: t[1][1]):
        items = list(items)
        start, end = items[0][0], items[-1][0]
        group_spans.append((grp, start, end))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([(s + e) / 2 for _, s, e in group_spans])
    ax2.set_xticklabels([g for g, _, _ in group_spans], fontsize=8, color="dimgrey")
    ax2.tick_params(length=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/plots_task_a_comparison")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    legend_handles = [
        mpatches.Patch(color=COLOR_PRETRAINED, label="Pretrained SAM"),
        mpatches.Patch(color=COLOR_FINETUNED,  label="Fine-tuned SAM (task_e)"),
        # mpatches.Patch(color=COLOR_NONE,        label="Not available"),
    ]

    # ---- Figure 1: mAP@50:95 (main comparison) ----
    fig, ax = plt.subplots(figsize=(16, 6))
    make_grouped_bar(ax, "map5095", "mAP @ 0.50:0.95", EXPERIMENTS)
    ax.set_title("Pretrained vs Fine-tuned SAM — mAP @ 0.50:0.95\n(all task_a prompt strategies)",
                 fontsize=13, pad=20)
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.92), fontsize=9)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "comparison_map5095.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ---- Figure 2: 2x2 — mAP50 / mAP75 / mAP_Ped / mAP_Car ----
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    sub_metrics = [
        ("map50",   "mAP @ 0.50"),
        ("map75",   "mAP @ 0.75"),
        ("map_ped", "mAP Pedestrian"),
        ("map_car", "mAP Car"),
    ]
    for ax, (key, label) in zip(axes.flat, sub_metrics):
        make_grouped_bar(ax, key, label, EXPERIMENTS)
        ax.set_title(label, fontsize=11)

    fig.suptitle("Pretrained vs Fine-tuned SAM — breakdown by metric",
                 fontsize=14, y=1.01)
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 1.00), fontsize=9)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "comparison_breakdown.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ---- Figure 3: mAP@50:95 only for GT BBox vs best point strategies ----
    highlight_keys = {"GT BBox", "Rand Mask n=3", "Mask Centroid", "BBox Center", "SIFT TopK n=3"}
    highlight_exps = [e for e in EXPERIMENTS if e[0] in highlight_keys]
    highlight_exps.sort(key=lambda e: (e[3] or {}).get("map5095", 0), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    make_grouped_bar(ax, "map5095", "mAP @ 0.50:0.95", highlight_exps)
    ax.set_title("Best strategies — Pretrained vs Fine-tuned SAM\nmAP @ 0.50:0.95",
                 fontsize=12, pad=20)
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.99, 0.92), fontsize=9)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "comparison_highlight.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
