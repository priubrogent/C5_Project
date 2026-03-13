"""
Task F — DeART (European Art) qualitative evaluation.

Runs SAM (pretrained and/or fine-tuned) on images from the
biglam/european_art dataset using the provided ground-truth person/nude
bounding boxes as prompts, and saves side-by-side qualitative figures.

No segmentation ground truth is available for this dataset, so only
qualitative results are produced (no mAP/mAR metrics).

Requirements
------------
  pip install datasets

The dataset is assumed to already be cached locally by HuggingFace datasets.
If not, the first run will download it automatically.

Usage
-----
# Pretrained baseline only (20 images):
python task_f/deart.py

# Compare pretrained vs fine-tuned:
python task_f/deart.py --checkpoint path/to/best_model_merged.pt

# Custom number of images / HF split:
python task_f/deart.py --num_images 40 --hf_split train
"""

from __future__ import annotations

import json
import os
import sys
import argparse

import numpy as np
from PIL import Image

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import load_sam, overlay_masks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEART_PERSON_CLASSES = {"person", "nude"}
PERSON_COLOR = (255, 120, 120)  # R,G,B for mask overlay


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_person_boxes(ann_str: str) -> tuple[list[list[float]], list[str]]:
    """Parse the JSON annotations string in a DeART item.

    Returns (boxes_xyxy, category_names) for person/nude annotations only.
    """
    ann_data = json.loads(ann_str)
    local_cat = {
        c["id"]: c["name"].lower()
        for c in ann_data.get("categories", [])
    }
    boxes: list[list[float]] = []
    names: list[str]         = []
    for ann in ann_data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        cat_name = local_cat.get(ann["category_id"], "")
        if cat_name not in DEART_PERSON_CLASSES:
            continue
        x, y, w, h = ann["bbox"]
        if w < 2 or h < 2:
            continue
        boxes.append([x, y, x + w, y + h])
        names.append(cat_name)
    return boxes, names


# ---------------------------------------------------------------------------
# Qualitative inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_qualitative(
    models: dict[str, tuple],   # {run_label: (model, processor)}
    hf_dataset,
    device: str,
    out_dir: str,
    num_images: int = 20,
) -> None:
    """Save side-by-side qualitative figures for `num_images` images.

    For each image a single figure is produced with:
      • Original image + GT bounding boxes
      • One SAM-segmented column per model variant
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for idx, item in enumerate(tqdm(hf_dataset, desc="DeART inference")):
        if saved >= num_images:
            break

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        boxes, names = _get_person_boxes(item["annotations"])
        if not boxes:
            continue  # skip images without person annotations

        run_labels = list(models.keys())
        n_cols     = 1 + len(run_labels)  # original + one per model
        fig, axes  = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

        # Column 0: original + GT boxes
        ax0 = axes[0] if n_cols > 1 else axes
        ax0.imshow(image)
        ax0.set_title("Original + GT boxes")
        ax0.axis("off")
        for (x1, y1, x2, y2), name in zip(boxes, names):
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle="square,pad=0",
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            ax0.add_patch(rect)
            ax0.text(x1, max(y1 - 4, 0), name, color="lime",
                     fontsize=8, va="bottom")

        # Columns 1+: SAM predictions per model
        for col_idx, run_label in enumerate(run_labels, start=1):
            model, processor = models[run_label]
            inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt")
            pv  = inputs["pixel_values"].to(device)
            ib  = inputs["input_boxes"].reshape(1, -1, 4).to(device)  # (1, N, 4)
            os_ = inputs["original_sizes"]
            ris = inputs["reshaped_input_sizes"]

            out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
            masks_list = processor.post_process_masks(
                masks=out.pred_masks,
                original_sizes=os_,
                reshaped_input_sizes=ris,
            )
            pred = masks_list[0].cpu().float()  # (N, 1, H, W)
            pred_bin = [(pred[k, 0].numpy() > 0) for k in range(pred.shape[0])]
            colors   = [PERSON_COLOR] * len(pred_bin)
            vis      = overlay_masks(image, pred_bin, colors)

            ax = axes[col_idx]
            ax.imshow(vis)
            ax.set_title(run_label.replace("_", " "))
            ax.axis("off")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"deart_{idx:05d}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close()
        saved += 1

    print(f"Saved {saved} qualitative images to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qualitative SAM evaluation on the DeART / European Art dataset."
    )
    p.add_argument("--model_id",   default="facebook/sam-vit-base",
                   help="HuggingFace model ID for the base SAM architecture.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to merged fine-tuned .pt state-dict (optional). "
                        "If provided, results from pretrained AND fine-tuned are shown "
                        "side-by-side in each figure.")
    p.add_argument("--hf_dataset", default="biglam/european_art",
                   help="HuggingFace dataset identifier for DeART.")
    p.add_argument("--hf_split",   default="train",
                   help="Dataset split to use (default: train).")
    p.add_argument("--output_dir",
                   default=os.path.join(
                       os.path.dirname(__file__), "..", "outputs", "task_f", "deart"
                   ),
                   help="Where to write qualitative figures.")
    p.add_argument("--num_images", type=int, default=20,
                   help="Number of images to process (among those with person annotations).")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model(s) ─────────────────────────────────────────────────────────
    models: dict[str, tuple] = {}

    print("\nLoading pretrained SAM…")
    model_pre, proc_pre = load_sam(args.model_id, checkpoint=None)
    model_pre = model_pre.to(device).eval()
    models["pretrained"] = (model_pre, proc_pre)

    if args.checkpoint:
        label = "finetuned_" + os.path.splitext(os.path.basename(args.checkpoint))[0]
        print(f"\nLoading fine-tuned SAM ({label})…")
        model_ft, proc_ft = load_sam(args.model_id, checkpoint=args.checkpoint)
        model_ft = model_ft.to(device).eval()
        models[label] = (model_ft, proc_ft)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"\nLoading dataset {args.hf_dataset} (split={args.hf_split})…")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for DeART evaluation.\n"
            "Install with:  pip install datasets"
        )

    hf_ds = load_dataset(args.hf_dataset, split=args.hf_split)
    print(f"  Dataset size: {len(hf_ds)} images")

    # ── Qualitative inference ─────────────────────────────────────────────────
    run_qualitative(
        models=models,
        hf_dataset=hf_ds,
        device=device,
        out_dir=args.output_dir,
        num_images=args.num_images,
    )

    print(f"\nDone. Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
