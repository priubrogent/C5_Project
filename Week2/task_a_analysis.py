import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from tqdm import tqdm

CAT_NAMES  = {1: "Pedestrian", 3: "Car"}
CAT_COLORS = {1: "#2ecc71", 3: "#3498db"}


def decode_rle(rle_dict, h, w):
    if isinstance(rle_dict, dict) and isinstance(rle_dict.get("counts"), str):
        rle_dict = {"counts": rle_dict["counts"].encode(), "size": rle_dict["size"]}
    return mask_util.decode(rle_dict).astype(np.uint8)


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def overlay_single(image, mask, color=(0, 255, 0), alpha=0.45):
    img = np.array(image).copy().astype(float)
    for c, v in enumerate(color):
        img[:, :, c] = np.where(mask, img[:, :, c] * (1 - alpha) + v * alpha, img[:, :, c])
    return img.astype(np.uint8)


def compute_per_ann_iou(coco_gt, predictions, dataset_root):
    pred_index = {}
    for p in predictions:
        key = (p["image_id"], p["category_id"])
        pred_index.setdefault(key, []).append(p)

    rows = []
    all_img_ids = list({p["image_id"] for p in predictions})

    for img_id in tqdm(all_img_ids, desc="Computing per-ann IoU"):
        img_info = coco_gt.loadImgs([img_id])[0]
        H, W = img_info["height"], img_info["width"]
        seq_id   = img_id // 100000
        frame_id = img_id % 100000

        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns    = coco_gt.loadAnns(ann_ids)
        anns    = [a for a in anns if a["category_id"] in CAT_NAMES
                   and a.get("iscrowd", 0) == 0]

        n_obj = len(anns)

        for ann in anns:
            cat_id  = ann["category_id"]
            bx, by, bw, bh = ann["bbox"]
            area    = bw * bh
            aspect  = bh / bw if bw > 0 else 1.0

            gt_mask = decode_rle(ann["segmentation"], H, W)
            gt_area_px = int(gt_mask.sum())

            preds_here = pred_index.get((img_id, cat_id), [])

            if not preds_here:
                rows.append(dict(
                    img_id=img_id, seq_id=seq_id, frame_id=frame_id,
                    file_name=img_info["file_name"],
                    cat_id=cat_id, bbox_area=area, aspect_ratio=aspect,
                    gt_area_px=gt_area_px, pred_area_px=0,
                    iou=0.0, sam_score=0.0, matched=False, n_obj=n_obj,
                ))
                continue

            best_iou   = -1.0
            best_score = 0.0
            best_pred_area = 0
            for p in preds_here:
                pm = decode_rle(p["segmentation"], H, W)
                iou = mask_iou(gt_mask, pm)
                if iou > best_iou:
                    best_iou       = iou
                    best_score     = p["score"]
                    best_pred_area = int(pm.sum())

            rows.append(dict(
                img_id=img_id, seq_id=seq_id, frame_id=frame_id,
                file_name=img_info["file_name"],
                cat_id=cat_id, bbox_area=area, aspect_ratio=aspect,
                gt_area_px=gt_area_px, pred_area_px=best_pred_area,
                iou=best_iou, sam_score=best_score, matched=True, n_obj=n_obj,
            ))

    return pd.DataFrame(rows)


def plot_iou_distribution(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    bins = np.linspace(0, 1, 21)
    for ax, (subset, title) in zip(axes, [
        (df, "All"),
        (df[df.cat_id == 1], "Pedestrian"),
        (df[df.cat_id == 3], "Car"),
    ]):
        ax.hist(subset["iou"], bins=bins, color="#3498db", edgecolor="white", linewidth=0.5)
        mean_iou = subset["iou"].mean()
        ax.axvline(mean_iou, color="red", linestyle="--", label=f"mean={mean_iou:.3f}")
        ax.set_title(title)
        ax.set_xlabel("IoU with GT")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    plt.suptitle("Per-annotation IoU distribution", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_distribution.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → iou_distribution.png")


def plot_iou_vs_area(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for cat_id, name in CAT_NAMES.items():
        sub = df[df.cat_id == cat_id]
        ax.scatter(sub["bbox_area"], sub["iou"], alpha=0.15, s=8,
                   color=CAT_COLORS[cat_id], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Bbox area (px²) — log scale")
    ax.set_ylabel("IoU")
    ax.set_title("IoU vs object size (scatter)")
    ax.legend()

    ax = axes[1]
    bins = np.percentile(df["bbox_area"], np.linspace(0, 100, 11))
    bins = np.unique(bins)
    labels = np.arange(len(bins) - 1)
    df["area_bin"] = pd.cut(df["bbox_area"], bins=bins, labels=labels, include_lowest=True)
    for cat_id, name in CAT_NAMES.items():
        sub = df[df.cat_id == cat_id].dropna(subset=["area_bin"])
        means = sub.groupby("area_bin", observed=True)["iou"].mean()
        ax.plot(means.index.astype(int), means.values, "o-",
                color=CAT_COLORS[cat_id], label=name)
    ax.set_xlabel("Area percentile bin (0=smallest → 9=largest)")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Mean IoU per size decile")
    ax.legend()
    ax.set_xticks(range(len(bins) - 1))

    plt.suptitle("IoU vs object area", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_vs_area.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → iou_vs_area.png")


def plot_iou_vs_aspect(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for cat_id, name in CAT_NAMES.items():
        sub = df[df.cat_id == cat_id]
        ax.scatter(sub["aspect_ratio"], sub["iou"], alpha=0.15, s=8,
                   color=CAT_COLORS[cat_id], label=name)
    ax.set_xlabel("Aspect ratio (h/w)")
    ax.set_ylabel("IoU")
    ax.set_title("IoU vs aspect ratio (scatter)")
    ax.legend()

    ax = axes[1]
    df["ar_bin"] = pd.cut(df["aspect_ratio"], bins=10, labels=False)
    bin_centers = df.groupby("ar_bin", observed=True)["aspect_ratio"].mean()
    for cat_id, name in CAT_NAMES.items():
        sub = df[df.cat_id == cat_id].dropna(subset=["ar_bin"])
        means = sub.groupby("ar_bin", observed=True)["iou"].mean()
        xs = [bin_centers.get(i, i) for i in means.index]
        ax.plot(xs, means.values, "o-", color=CAT_COLORS[cat_id], label=name)
    ax.set_xlabel("Mean aspect ratio in bin")
    ax.set_ylabel("Mean IoU")
    ax.set_title("Mean IoU per aspect ratio bin")
    ax.legend()

    plt.suptitle("IoU vs aspect ratio (h/w)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_vs_aspectratio.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → iou_vs_aspectratio.png")


def plot_iou_vs_nobj(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby("n_obj")["iou"].mean()
    counts = df.groupby("n_obj")["iou"].count()
    ax.bar(means.index, means.values, color="#9b59b6", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(counts.index, counts.values, "o--", color="gray", alpha=0.6, label="# instances")
    ax.set_xlabel("Objects in image")
    ax.set_ylabel("Mean IoU")
    ax2.set_ylabel("# instances (gray)")
    ax.set_title("Does crowding hurt SAM? (IoU vs objects per image)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_vs_nobj.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → iou_vs_nobj.png")


def plot_per_sequence(df, out_dir):
    seq_stats = df.groupby("seq_id").agg(
        mean_iou=("iou", "mean"),
        iou_50=("iou", lambda x: (x >= 0.5).mean()),
        iou_75=("iou", lambda x: (x >= 0.75).mean()),
        count=("iou", "count"),
    ).reset_index().sort_values("mean_iou")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.barh(seq_stats["seq_id"].astype(str), seq_stats["mean_iou"],
                   color="#1abc9c", edgecolor="white")
    ax.set_xlabel("Mean IoU")
    ax.set_title("Mean IoU per sequence")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, seq_stats["mean_iou"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax = axes[1]
    x = np.arange(len(seq_stats))
    w = 0.35
    ax.bar(x - w/2, seq_stats["iou_50"], w, label="IoU≥0.5", color="#e74c3c")
    ax.bar(x + w/2, seq_stats["iou_75"], w, label="IoU≥0.75", color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_stats["seq_id"].astype(str), rotation=45)
    ax.set_ylabel("Fraction of instances")
    ax.set_title("Success rate per sequence")
    ax.legend()

    plt.suptitle("Per-sequence performance", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_sequence.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → per_sequence.png")


def plot_sam_calibration(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(df["sam_score"], df["iou"], alpha=0.1, s=6, color="#e74c3c")
    bins = np.linspace(0, 1, 21)
    labels = (bins[:-1] + bins[1:]) / 2
    df["score_bin"] = pd.cut(df["sam_score"], bins=bins, labels=labels)
    means = df.groupby("score_bin", observed=True)["iou"].mean()
    ax.plot(means.index.astype(float), means.values, "b-o", linewidth=2, markersize=5,
            label="Binned mean IoU")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("SAM predicted IOU score")
    ax.set_ylabel("Actual IoU with GT")
    ax.set_title("SAM score calibration")
    ax.legend()

    ax = axes[1]
    bins_iou = np.linspace(0, 1, 11)
    for cat_id, name in CAT_NAMES.items():
        sub = df[df.cat_id == cat_id]
        ax.hist(sub["sam_score"], bins=bins_iou, alpha=0.6, label=name,
                color=CAT_COLORS[cat_id])
    ax.set_xlabel("SAM predicted IOU score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of SAM confidence scores")
    ax.legend()

    plt.suptitle("SAM self-assessment calibration", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sam_score_vs_iou.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → sam_score_vs_iou.png")


def plot_segmentation_ratio(df, out_dir):
    valid = df[(df.gt_area_px > 0) & (df.pred_area_px > 0)].copy()
    valid["seg_ratio"] = valid["pred_area_px"] / valid["gt_area_px"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(0, 4, 41)
    for cat_id, name in CAT_NAMES.items():
        sub = valid[valid.cat_id == cat_id]
        ax.hist(np.clip(sub["seg_ratio"], 0, 4), bins=bins, alpha=0.6,
                label=name, color=CAT_COLORS[cat_id])
    ax.axvline(1.0, color="black", linestyle="--", label="Perfect (ratio=1)")
    ax.set_xlabel("pred_area / gt_area  (clipped at 4)")
    ax.set_ylabel("Count")
    ax.set_title("Over- vs Under-segmentation")
    ax.legend()

    ax = axes[1]
    ax.scatter(valid["iou"], valid["seg_ratio"], alpha=0.1, s=6, color="#8e44ad")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("IoU")
    ax.set_ylabel("pred_area / gt_area")
    ax.set_ylim(0, 5)
    pct_over  = (valid["seg_ratio"] > 1.1).mean() * 100
    pct_under = (valid["seg_ratio"] < 0.9).mean() * 100
    ax.set_title(f"Seg ratio vs IoU  (over={pct_over:.0f}%, under={pct_under:.0f}%)")

    plt.suptitle("Segmentation area analysis", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "segmentation_ratio.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  → segmentation_ratio.png")


def plot_frame_progression(df, out_dir):
    seq_ids = sorted(df["seq_id"].unique())
    n = len(seq_ids)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.5))
    axes = axes.flatten()

    for i, seq_id in enumerate(seq_ids):
        sub = df[df.seq_id == seq_id].sort_values("frame_id")
        frame_means = sub.groupby("frame_id")["iou"].mean()
        axes[i].plot(frame_means.index, frame_means.values, alpha=0.7, linewidth=0.8)
        if len(frame_means) > 5:
            rolling = frame_means.rolling(10, center=True).mean()
            axes[i].plot(rolling.index, rolling.values, "r-", linewidth=2)
        axes[i].set_title(f"Seq {seq_id}", fontsize=9)
        axes[i].set_xlabel("Frame", fontsize=7)
        axes[i].set_ylabel("Mean IoU", fontsize=7)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("IoU progression over frames (red = rolling mean)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "frame_progression.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("  → frame_progression.png")


def plot_montage(df, coco_gt, predictions, dataset_root, out_dir, mode="failure", n=12):
    assert mode in ("failure", "success")
    df_sorted = df[df.matched].sort_values("iou", ascending=(mode == "failure"))
    top = df_sorted.head(n)

    pred_index = {}
    for p in predictions:
        key = (p["image_id"], p["category_id"])
        pred_index.setdefault(key, []).append(p)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 3.5))

    fname_base = f"{'failure' if mode == 'failure' else 'success'}_montage"
    txt_lines = [f"# {mode.upper()} cases — image paths for manual inspection\n",
                 f"# Columns: rank | actual_IoU | SAM_score | category | img_path\n\n"]

    for idx, (_, row) in enumerate(top.iterrows()):
        r = idx // cols
        c = (idx % cols) * 2

        img_info = coco_gt.loadImgs([int(row.img_id)])[0]
        H, W = img_info["height"], img_info["width"]
        img_path = os.path.join(dataset_root, img_info["file_name"])

        txt_lines.append(
            f"{idx+1:>3}  IoU={row.iou:.3f}  score={row.sam_score:.3f}"
            f"  {CAT_NAMES.get(int(row.cat_id), '?'):>11}  {img_path}\n"
        )

        if not os.path.exists(img_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = coco_gt.getAnnIds(imgIds=[int(row.img_id)])
        anns = coco_gt.loadAnns(ann_ids)
        anns_cat = [a for a in anns if a["category_id"] == int(row.cat_id)
                    and a.get("iscrowd", 0) == 0]
        if not anns_cat:
            continue
        anns_cat.sort(key=lambda a: abs(a["bbox"][2] * a["bbox"][3] - row.bbox_area))
        ann = anns_cat[0]

        gt_mask = mask_util.decode(ann["segmentation"]).astype(np.uint8)

        preds = pred_index.get((int(row.img_id), int(row.cat_id)), [])
        if not preds:
            continue
        best_p = max(preds, key=lambda p: mask_iou(
            gt_mask, mask_util.decode({"counts": p["segmentation"]["counts"].encode(),
                                       "size": p["segmentation"]["size"]})))
        pm = mask_util.decode({"counts": best_p["segmentation"]["counts"].encode(),
                               "size": best_p["segmentation"]["size"]}).astype(np.uint8)

        bx, by, bw, bh = [int(v) for v in ann["bbox"]]
        pad = 20
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(W, bx + bw + pad), min(H, by + bh + pad)

        gt_vis   = overlay_single(image[y1:y2, x1:x2], gt_mask[y1:y2, x1:x2], (0, 200, 0))
        pred_vis = overlay_single(image[y1:y2, x1:x2], pm[y1:y2, x1:x2], (200, 50, 50))

        ax_gt   = axes[r, c]
        ax_pred = axes[r, c + 1]
        ax_gt.imshow(gt_vis)
        ax_gt.axis("off")
        cat_name = CAT_NAMES.get(int(row.cat_id), "?")
        ax_gt.set_title(f"GT  [{cat_name}]\n{img_info['file_name']}", fontsize=6)
        ax_pred.imshow(pred_vis)
        ax_pred.axis("off")
        ax_pred.set_title(f"Pred  actual IoU={row.iou:.3f}\nSAM score={row.sam_score:.3f}",
                          fontsize=6)

    for j in range(idx + 1, rows * cols):
        r2, c2 = j // cols, (j % cols) * 2
        axes[r2, c2].set_visible(False)
        axes[r2, c2 + 1].set_visible(False)

    title = (f"{'Worst' if mode == 'failure' else 'Best'} {n} predictions  "
             f"(left=GT mask [green] | right=SAM prediction [red])\n"
             f"'actual IoU' = overlap with GT mask;  'SAM score' = model's own confidence")
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname_base + ".png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → {fname_base}.png")

    txt_path = os.path.join(out_dir, fname_base + "_paths.txt")
    with open(txt_path, "w") as f:
        f.writelines(txt_lines)
    print(f"  → {fname_base}_paths.txt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir",
                   default="/hhome/priubrogent/mcvpol/C5/Week2/outputs/task_a_experiments")
    p.add_argument("--strategy",
                   choices=["bbox_center", "mask_centroid", "random_mask", "random_bbox"],
                   default="bbox_center")
    p.add_argument("--num_points", type=int, default=1)
    p.add_argument("--neg_points", type=int, default=0)
    p.add_argument("--dataset_root",
                   default="/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--montage_n", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()

    suffix = args.strategy
    if args.num_points > 1:
        suffix += f"_n{args.num_points}"
    if args.neg_points > 0:
        suffix += f"_neg{args.neg_points}"
    results_dir = os.path.join(args.experiments_dir, suffix)

    out_dir = args.output_dir or os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data from {results_dir}")
    coco_gt = COCO(os.path.join(results_dir, "gt_fixed.json"))

    with open(os.path.join(results_dir, "predictions.json")) as f:
        predictions = json.load(f)

    print(f"GT images: {len(coco_gt.getImgIds())}  |  Predictions: {len(predictions)}")

    cache_path = os.path.join(out_dir, "per_ann_stats.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached per-ann stats from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        df = compute_per_ann_iou(coco_gt, predictions, args.dataset_root)
        df.to_csv(cache_path, index=False)
        print(f"  → per_ann_stats.csv  ({len(df)} annotations)")

    print(f"\nOverall mean IoU : {df['iou'].mean():.4f}")
    print(f"IoU≥0.5 rate    : {(df['iou'] >= 0.5).mean():.4f}")
    print(f"IoU≥0.75 rate   : {(df['iou'] >= 0.75).mean():.4f}")
    for cat_id, name in CAT_NAMES.items():
        sub = df[df.cat_id == cat_id]
        print(f"  {name}: mean IoU={sub['iou'].mean():.4f}  n={len(sub)}")

    print("\nGenerating plots...")
    plot_iou_distribution(df, out_dir)
    plot_iou_vs_area(df, out_dir)
    plot_iou_vs_aspect(df, out_dir)
    plot_iou_vs_nobj(df, out_dir)
    plot_per_sequence(df, out_dir)
    plot_sam_calibration(df, out_dir)
    plot_segmentation_ratio(df, out_dir)
    plot_frame_progression(df, out_dir)
    plot_montage(df, coco_gt, predictions, args.dataset_root, out_dir,
                 mode="failure", n=args.montage_n)
    plot_montage(df, coco_gt, predictions, args.dataset_root, out_dir,
                 mode="success", n=args.montage_n)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
