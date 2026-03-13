"""
Task F — HoofedAnimals dataset evaluation.

Evaluates pretrained SAM (and optionally a fine-tuned checkpoint) on the
HoofedAnimals segmentation benchmark using ground-truth bounding boxes
derived from the provided mask files as prompts.

Dataset layout
--------------
<root>/org/<id>.pgm              grayscale images, id ∈ {1..200}
<root>/seg/masks/<id>_mask.ppm   RGB color-coded instance masks

Mask color convention (Borji et al., 2012)
------------------------------------------
Each animal class has a base (R,G,B) color.  Additional instances of the
same class use decreasing intensities, stepping by 10 per instance:

    cow    (255, 0,   0)   — red
    horse  (0,   255, 0)   — green
    sheep  (0,   0,   255) — blue
    goat   (255, 0,   255) — magenta
    camel  (255, 255, 0)   — yellow
    deer   (0,   255, 255) — cyan

Example: two sheep → (0,0,255) and (0,0,245).
Background pixels are black (0,0,0).

Metrics
-------
COCO segmentation mAP / mAR reported overall and per category.

Usage
-----
# Pretrained baseline only:
python task_f/hoofed_animals.py

# Compare pretrained vs fine-tuned:
python task_f/hoofed_animals.py --checkpoint path/to/best_model_merged.pt
"""

from __future__ import annotations

import json
import os
import sys
import argparse
import tempfile

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import SamProcessor

sys.path.insert(0, os.path.dirname(__file__))
from common import load_sam, mask_to_rle, overlay_masks, sam_predict_boxes


# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    {"id": 1, "name": "cow"},
    {"id": 2, "name": "horse"},
    {"id": 3, "name": "sheep"},
    {"id": 4, "name": "goat"},
    {"id": 5, "name": "camel"},
    {"id": 6, "name": "deer"},
]
CAT_NAMES = {c["id"]: c["name"] for c in CATEGORIES}

# Maps (r>0, g>0, b>0) channel-presence pattern → category_id
_PATTERN_TO_CAT: dict[tuple[bool, bool, bool], int] = {
    (True,  False, False): 1,  # cow    — red
    (False, True,  False): 2,  # horse  — green
    (False, False, True):  3,  # sheep  — blue
    (True,  False, True):  4,  # goat   — magenta
    (True,  True,  False): 5,  # camel  — yellow
    (False, True,  True):  6,  # deer   — cyan
}

# Visualization colors (R, G, B 0-255)
VIS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (220,  60,  60),   # cow
    2: ( 60, 180,  60),   # horse
    3: ( 60, 100, 220),   # sheep
    4: (200,  60, 200),   # goat
    5: (220, 200,  60),   # camel
    6: ( 60, 200, 200),   # deer
}


# ---------------------------------------------------------------------------
# Mask parsing
# ---------------------------------------------------------------------------

def parse_mask_file(mask_path: str) -> list[tuple[int, np.ndarray]]:
    """Parse a HoofedAnimals *_mask.ppm file into per-instance binary masks.

    Returns list of (category_id, H×W bool array) tuples, one per instance.
    """
    arr = np.array(Image.open(mask_path))  # (H, W, 3) uint8
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)

    instances: list[tuple[int, np.ndarray]] = []
    for color in unique_colors:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        if r == 0 and g == 0 and b == 0:
            continue  # background

        pattern = (r > 0, g > 0, b > 0)
        cat_id = _PATTERN_TO_CAT.get(pattern)
        if cat_id is None:
            continue  # unexpected pattern — skip

        binary = np.all(arr == color[None, None, :], axis=2)
        if binary.sum() < 16:
            continue  # discard tiny noise regions

        instances.append((cat_id, binary))

    return instances


def _mask_bbox_xywh(mask: np.ndarray) -> list[float]:
    """Return [x, y, w, h] (COCO-style) bounding box of a binary mask."""
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    y1, y2 = int(rows[0]), int(rows[-1])
    x1, x2 = int(cols[0]), int(cols[-1])
    return [x1, y1, x2 - x1, y2 - y1]


def _mask_bbox_xyxy(mask: np.ndarray) -> list[float]:
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    y1, y2 = int(rows[0]), int(rows[-1])
    x1, x2 = int(cols[0]), int(cols[-1])
    return [float(x1), float(y1), float(x2), float(y2)]


# ---------------------------------------------------------------------------
# COCO GT builder
# ---------------------------------------------------------------------------

def build_coco_gt(root: str, out_dir: str) -> tuple[COCO, dict[int, list[tuple[int, np.ndarray]]]]:
    """Parse all mask files, write a COCO GT JSON, and return a COCO object.

    Also returns a dict mapping image_id → [(cat_id, binary_mask)] for later
    use in qualitative visualization without re-parsing.
    """
    org_dir  = os.path.join(root, "org")
    mask_dir = os.path.join(root, "seg", "masks")

    image_ids = sorted(
        int(f.replace(".pgm", ""))
        for f in os.listdir(org_dir)
        if f.endswith(".pgm")
    )

    coco_images: list[dict]      = []
    coco_anns:   list[dict]      = []
    instances:   dict[int, list] = {}
    ann_id = 1

    for img_id in tqdm(image_ids, desc="Building GT"):
        img_path  = os.path.join(org_dir,  f"{img_id}.pgm")
        mask_path = os.path.join(mask_dir, f"{img_id}_mask.ppm")
        if not os.path.exists(mask_path):
            continue

        inst_list = parse_mask_file(mask_path)
        if not inst_list:
            continue

        img        = Image.open(img_path)
        W, H       = img.size
        instances[img_id] = inst_list
        coco_images.append({"id": img_id, "file_name": f"{img_id}.pgm", "height": H, "width": W})

        for cat_id, mask in inst_list:
            coco_anns.append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  cat_id,
                "segmentation": mask_to_rle(mask),
                "bbox":         _mask_bbox_xywh(mask),
                "area":         int(mask.sum()),
                "iscrowd":      0,
            })
            ann_id += 1

    gt_dict = {"images": coco_images, "annotations": coco_anns, "categories": CATEGORIES}
    gt_path = os.path.join(out_dir, "hoofed_gt.json")
    with open(gt_path, "w") as f:
        json.dump(gt_dict, f)

    coco_gt = COCO(gt_path)
    return coco_gt, instances


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HoofedAnimalsDataset(Dataset):
    """Returns SAM-ready tensors for images that have at least one instance."""

    def __init__(
        self,
        root: str,
        instances: dict[int, list[tuple[int, np.ndarray]]],
        processor: SamProcessor,
    ) -> None:
        self.org_dir   = os.path.join(root, "org")
        self.instances = instances
        self.processor = processor
        self.image_ids = sorted(instances.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id    = self.image_ids[idx]
        image     = Image.open(os.path.join(self.org_dir, f"{img_id}.pgm")).convert("RGB")
        inst_list = self.instances[img_id]

        cat_ids = [c for c, _ in inst_list]
        masks   = [m for _, m in inst_list]
        boxes   = [_mask_bbox_xyxy(m) for m in masks]

        inputs = self.processor(images=image, input_boxes=[boxes], return_tensors="pt")

        return {
            "pixel_values":         inputs["pixel_values"].squeeze(0),
            "input_boxes":          inputs["input_boxes"].reshape(-1, 4),  # (N, 4)
            "original_sizes":       inputs["original_sizes"].squeeze(0),
            "reshaped_input_sizes": inputs["reshaped_input_sizes"].squeeze(0),
            "cat_ids":              torch.tensor(cat_ids, dtype=torch.long),
            "gt_masks":             masks,   # list of H×W bool arrays
            "img_id":               img_id,
        }


# ---------------------------------------------------------------------------
# Inference + COCO evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    dataset: HoofedAnimalsDataset,
    processor: SamProcessor,
    device: str,
    coco_gt: COCO,
    out_dir: str,
    qual_images: int = 5,
) -> dict[str, float]:
    model.eval()
    results:    list[dict] = []
    qual_count: int        = 0
    qual_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Inference"):
        item    = dataset[idx]
        img_id  = item["img_id"]
        cat_ids = item["cat_ids"].tolist()

        pv  = item["pixel_values"].unsqueeze(0).to(device)
        ib  = item["input_boxes"].unsqueeze(0).to(device)  # (1, N, 4)
        os_ = item["original_sizes"].unsqueeze(0)
        ris = item["reshaped_input_sizes"].unsqueeze(0)

        out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
        masks_list = processor.post_process_masks(
            masks=out.pred_masks,
            original_sizes=os_,
            reshaped_input_sizes=ris,
        )
        pred_masks = masks_list[0].cpu().float()  # (N, 1, H, W)

        for k, cat_id in enumerate(cat_ids):
            mask_np = (pred_masks[k, 0].numpy() > 0).astype(np.uint8)
            results.append({
                "image_id":     img_id,
                "category_id":  cat_id,
                "segmentation": mask_to_rle(mask_np),
                "score":        1.0,
            })

        if qual_count < qual_images:
            image = Image.open(
                os.path.join(dataset.org_dir, f"{img_id}.pgm")
            ).convert("RGB")

            colors    = [VIS_COLORS[c] for c in cat_ids]
            pred_bin  = [(pred_masks[k, 0].numpy() > 0) for k in range(len(cat_ids))]
            gt_vis    = overlay_masks(image, item["gt_masks"], colors)
            pred_vis  = overlay_masks(image, pred_bin, colors)

            present_cats = sorted(set(cat_ids))
            legend = [
                mpatches.Patch(
                    color=[v / 255 for v in VIS_COLORS[cid]],
                    label=CAT_NAMES[cid],
                )
                for cid in present_cats
            ]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(gt_vis);   axes[0].set_title("GT masks");      axes[0].axis("off")
            axes[1].imshow(pred_vis); axes[1].set_title("SAM predicted"); axes[1].axis("off")
            fig.legend(handles=legend, loc="lower center", ncol=len(legend), fontsize=10)
            plt.tight_layout()
            plt.savefig(
                os.path.join(qual_dir, f"img_{img_id:04d}.png"),
                dpi=100, bbox_inches="tight",
            )
            plt.close()
            qual_count += 1

    if not results:
        print("WARNING: no predictions generated.")
        return {}

    pred_coco = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, pred_coco, "segm")
    ev.params.imgIds = sorted({r["image_id"] for r in results})
    ev.params.catIds = [c["id"] for c in CATEGORIES]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75",
        "mAP_small", "mAP_medium", "mAP_large",
        "mAR_1", "mAR_10", "mAR_100",
        "mAR_small", "mAR_medium", "mAR_large",
    ]
    metrics = {n: round(float(v), 4) for n, v in zip(metric_names, ev.stats)}

    # Per-category breakdown
    precision = ev.eval["precision"]  # (T, R, K, A, M)
    recall    = ev.eval["recall"]     # (T, K, A, M)
    for k, cat in enumerate(sorted(CATEGORIES, key=lambda c: c["id"])):
        p = precision[:, :, k, 0, 2]
        r = recall[:, k, 0, 2]
        metrics[f"mAP_{cat['name']}"] = round(
            float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4
        )
        metrics[f"mAR_{cat['name']}"] = round(
            float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4
        )

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate SAM (pretrained and/or fine-tuned) on HoofedAnimals."
    )
    p.add_argument("--model_id",     default="facebook/sam-vit-base",
                   help="HuggingFace model ID for the base SAM architecture.")
    p.add_argument("--checkpoint",   default=None,
                   help="Path to merged fine-tuned .pt state-dict (optional). "
                        "If omitted only the pretrained model is evaluated.")
    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/HoofedAnimals/HoofedAnimals",
                   help="Root directory of the HoofedAnimals dataset.")
    p.add_argument("--output_dir",
                   default=os.path.join(
                       os.path.dirname(__file__), "..", "outputs", "task_f", "hoofed_animals"
                   ),
                   help="Where to write metrics JSON and qualitative images.")
    p.add_argument("--qual_images",  type=int, default=5,
                   help="Number of qualitative side-by-side images to save per run.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Build GT once (shared across both runs) ──────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print("\nParsing mask files and building COCO GT…")
    coco_gt, instances = build_coco_gt(args.dataset_root, args.output_dir)
    print(f"  {len(instances)} images with annotations")

    all_metrics: dict[str, dict] = {}

    # ── Runs to evaluate ─────────────────────────────────────────────────────
    runs: list[tuple[str, str | None]] = [("pretrained", None)]
    if args.checkpoint:
        label = "finetuned_" + os.path.splitext(os.path.basename(args.checkpoint))[0]
        runs.append((label, args.checkpoint))

    for run_label, ckpt in runs:
        print(f"\n{'='*60}")
        print(f"Run: {run_label}")
        print(f"{'='*60}")

        run_dir = os.path.join(args.output_dir, run_label)
        os.makedirs(run_dir, exist_ok=True)

        model, processor = load_sam(args.model_id, ckpt)
        model = model.to(device).eval()

        dataset = HoofedAnimalsDataset(args.dataset_root, instances, processor)

        metrics = run_inference(
            model, dataset, processor, device, coco_gt, run_dir, args.qual_images
        )

        all_metrics[run_label] = metrics

        result = {"run_label": run_label, "args": vars(args), "metrics": metrics}
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved → {metrics_path}")

    # ── Comparison table ──────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("Comparison")
        print(f"{'='*60}")
        header_keys = ["mAP_0.50_0.95", "mAP_0.50", "mAP_0.75",
                       "mAR_100"] + [f"mAP_{c['name']}" for c in CATEGORIES]
        col_w = max(len(k) for k in header_keys) + 2
        run_labels = list(all_metrics.keys())

        # Header
        print(f"{'metric':<{col_w}}" + "".join(f"{r:>12}" for r in run_labels))
        print("-" * (col_w + 12 * len(run_labels)))
        for k in header_keys:
            row = f"{k:<{col_w}}"
            for r in run_labels:
                v = all_metrics[r].get(k, float("nan"))
                row += f"{v:>12.4f}"
            print(row)

    # Save combined summary
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
