"""
Proper qualitative comparison between two segmentation methods.

For each validation image, matches predictions to GT using mask IoU and
classifies each prediction/GT as:
  TP (green)  — prediction matched to a GT instance (IoU > threshold)
  FP (red)    — prediction with no matching GT
  FN (yellow) — GT instance with no matching prediction

Ranks images by how differently the two methods behave and saves
side-by-side comparison panels.

Usage:
    python analyze_qualitative.py \
        --pred_a  outputs/task_c_arnau/original_thr0.25/predictions_segm.json \
        --pred_b  outputs/task_b_dinolarge/car_person_thr0.30/predictions.json \
        --gt      outputs/task_c_arnau/original_thr0.25/gt_fixed.json \
        --label_a "YOLO+SAM-base" \
        --label_b "DINO-tiny+SAM-large" \
        --dataset_root /home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02 \
        --output_dir outputs/qual_analysis \
        --top 20 \
        --iou_threshold 0.5
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_a",       required=True, help="Predictions JSON for method A")
    p.add_argument("--pred_b",       required=True, help="Predictions JSON for method B")
    p.add_argument("--gt",           required=True, help="GT COCO JSON (gt_fixed.json)")
    p.add_argument("--label_a",      default="Method A")
    p.add_argument("--label_b",      default="Method B")
    p.add_argument("--dataset_root", required=True, help="Path to KITTI-MOTS image_02 folder")
    p.add_argument("--output_dir",   default="outputs/qual_analysis")
    p.add_argument("--top",          type=int,   default=20,  help="Frames to save")
    p.add_argument("--iou_threshold",type=float, default=0.5, help="IoU to count as TP")
    p.add_argument("--cat_ids",      type=int, nargs="+", default=[1, 3],
                   help="Category IDs to evaluate (default: 1=Pedestrian 3=Car)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def rle_to_mask(rle, h, w):
    if isinstance(rle["counts"], list):
        rle = mask_util.frPyObjects(rle, h, w)
    return mask_util.decode(rle).astype(bool)


def batch_iou(gt_rle, pred_rles):
    """IoU between one GT RLE and a list of pred RLEs. Returns array of shape (N,)."""
    if not pred_rles:
        return np.array([])
    return mask_util.iou([gt_rle], pred_rles, [False])[0]


# ---------------------------------------------------------------------------
# Matching: greedy IoU matching (GT → best pred, one-to-one)
# ---------------------------------------------------------------------------

def match(gt_anns, preds, iou_thr):
    """
    Returns:
      tp_gt_idxs   : set of GT indices that were matched
      tp_pred_idxs : set of pred indices that matched a GT
      iou_per_gt   : dict {gt_idx: best_iou} for matched GTs
    """
    if not gt_anns or not preds:
        return set(), set(), {}

    # Encode all GT and pred masks as RLE
    gt_rles  = []
    for a in gt_anns:
        seg = a["segmentation"]
        if isinstance(seg["counts"], list):
            seg = mask_util.frPyObjects(seg, seg["size"][0], seg["size"][1])
        gt_rles.append(seg)

    pred_rles = []
    for pr in preds:
        seg = pr["segmentation"]
        if isinstance(seg["counts"], list):
            seg = mask_util.frPyObjects(seg, seg["size"][0], seg["size"][1])
        pred_rles.append(seg)

    # iou matrix: (n_gt, n_pred)
    iou_mat = mask_util.iou(gt_rles, pred_rles, [False] * len(pred_rles))
    if iou_mat.ndim == 1:
        iou_mat = iou_mat.reshape(1, -1)

    tp_gt, tp_pred, iou_per_gt = set(), set(), {}

    # Sort GT by max achievable IoU (descending) for greedy matching
    best_per_gt = iou_mat.max(axis=1) if iou_mat.size else np.array([])
    for gt_idx in np.argsort(-best_per_gt):
        row = iou_mat[gt_idx]
        # Only consider unmatched preds
        available = [(iou, pi) for pi, iou in enumerate(row) if pi not in tp_pred]
        if not available:
            break
        best_iou, best_pi = max(available, key=lambda x: x[0])
        if best_iou >= iou_thr:
            tp_gt.add(gt_idx)
            tp_pred.add(best_pi)
            iou_per_gt[gt_idx] = best_iou

    return tp_gt, tp_pred, iou_per_gt


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

COLORS = {
    "tp":      (0,   200,  0,   120),   # green  — correct detection
    "fp":      (220, 0,    0,   120),   # red    — false positive
    "fn":      (255, 220,  0,   120),   # yellow — missed GT
    "gt_only": (255, 255, 255,  80),    # white  — GT outline
}
CAT_COLORS = {1: (0, 255, 0), 3: (0, 0, 255)}


def draw_mask(draw_layer, mask_np, rgba):
    """Paint a boolean mask onto an RGBA layer."""
    overlay = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
    overlay[mask_np] = rgba
    return overlay


def render_panel(orig_img, gt_anns, pred_anns, tp_gt, tp_pred, img_info, label):
    """Return a PIL Image with colour-coded TP/FP/FN overlays."""
    h, w = img_info["height"], img_info["width"]
    base  = orig_img.copy().convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer_arr = np.zeros((h, w, 4), dtype=np.uint8)

    # TPs and FNs from GT side
    for gi, ann in enumerate(gt_anns):
        mask = rle_to_mask(ann["segmentation"], h, w)
        if gi in tp_gt:
            layer_arr[mask] = COLORS["tp"]
        else:
            layer_arr[mask] = COLORS["fn"]

    # FPs from pred side
    for pi, pred in enumerate(pred_anns):
        if pi not in tp_pred:
            mask = rle_to_mask(pred["segmentation"], h, w)
            layer_arr[mask] = COLORS["fp"]

    layer = Image.fromarray(layer_arr, "RGBA")
    out   = Image.alpha_composite(base, layer).convert("RGB")

    # Add legend text
    draw  = ImageDraw.Draw(out)
    draw.text((4, 4),  label,             fill=(255, 255, 255))
    draw.text((4, 18), "■ TP  ■ FP  ■ FN", fill=(200, 200, 200))
    draw.text((4, 32), "green red yellow", fill=(200, 200, 200))

    return out


def render_gt_panel(orig_img, gt_anns, img_info):
    """GT masks as white semi-transparent overlay."""
    h, w  = img_info["height"], img_info["width"]
    base  = orig_img.copy().convert("RGBA")
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    for ann in gt_anns:
        mask = rle_to_mask(ann["segmentation"], h, w)
        cat  = ann.get("category_id", 1)
        c    = CAT_COLORS.get(cat, (255, 255, 255))
        layer[mask] = (*c, 100)
    out = Image.alpha_composite(base, Image.fromarray(layer, "RGBA")).convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.text((4, 4), "Ground Truth", fill=(255, 255, 255))
    return out


def pad_to_height(img, target_h):
    if img.height == target_h:
        return img
    canvas = Image.new("RGB", (img.width, target_h), (20, 20, 20))
    canvas.paste(img, (0, 0))
    return canvas


def build_comparison(orig_img, img_info, gt_anns,
                     preds_a, tp_gt_a, tp_pred_a,
                     preds_b, tp_gt_b, tp_pred_b,
                     label_a, label_b, stats):
    gt_panel = render_gt_panel(orig_img, gt_anns, img_info)
    pan_a    = render_panel(orig_img, gt_anns, preds_a, tp_gt_a, tp_pred_a, img_info, label_a)
    pan_b    = render_panel(orig_img, gt_anns, preds_b, tp_gt_b, tp_pred_b, img_info, label_b)

    # Stats bar at bottom
    stat_h  = 30
    gap     = 4
    max_h   = max(gt_panel.height, pan_a.height, pan_b.height)
    gt_panel = pad_to_height(gt_panel, max_h)
    pan_a    = pad_to_height(pan_a,    max_h)
    pan_b    = pad_to_height(pan_b,    max_h)

    total_w = gt_panel.width + gap + pan_a.width + gap + pan_b.width
    canvas  = Image.new("RGB", (total_w, max_h + stat_h), (20, 20, 20))
    canvas.paste(gt_panel, (0, 0))
    canvas.paste(pan_a,    (gt_panel.width + gap, 0))
    canvas.paste(pan_b,    (gt_panel.width + gap + pan_a.width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    stat_txt = (
        f"{label_a}: TP={stats['tp_a']} FP={stats['fp_a']} FN={stats['fn_a']}    "
        f"{label_b}: TP={stats['tp_b']} FP={stats['fp_b']} FN={stats['fn_b']}    "
        f"GT={stats['n_gt']}"
    )
    draw.text((4, max_h + 6), stat_txt, fill=(220, 220, 220))
    return canvas


# ---------------------------------------------------------------------------
# Interest score
# ---------------------------------------------------------------------------

def interest_score(tp_gt_a, tp_gt_b, fp_a, fp_b, n_gt):
    """
    High when methods disagree most:
      - instances caught by one but not the other
      - large FP difference
    """
    only_a   = len(tp_gt_a - tp_gt_b)
    only_b   = len(tp_gt_b - tp_gt_a)
    fp_diff  = abs(fp_a - fp_b)
    return only_a + only_b + 0.5 * fp_diff


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GT...")
    coco_gt = COCO(args.gt)

    print("Loading predictions...")
    with open(args.pred_a) as f:
        raw_a = json.load(f)
    with open(args.pred_b) as f:
        raw_b = json.load(f)

    cat_ids = set(args.cat_ids)

    # Group predictions by image_id
    preds_a_by_img = {}
    for p in raw_a:
        if p["category_id"] in cat_ids:
            preds_a_by_img.setdefault(p["image_id"], []).append(p)

    preds_b_by_img = {}
    for p in raw_b:
        if p["category_id"] in cat_ids:
            preds_b_by_img.setdefault(p["image_id"], []).append(p)

    # Evaluate on images present in both prediction sets (or either)
    all_img_ids = sorted(
        set(preds_a_by_img) | set(preds_b_by_img)
    )
    # Filter to images that exist on disk
    valid_img_ids = []
    for iid in all_img_ids:
        info = coco_gt.loadImgs([iid])
        if not info:
            continue
        path = os.path.join(args.dataset_root, info[0]["file_name"])
        if os.path.exists(path):
            valid_img_ids.append(iid)

    print(f"Evaluating {len(valid_img_ids)} images...")

    scores = []
    for img_id in tqdm(valid_img_ids):
        img_info = coco_gt.loadImgs([img_id])[0]

        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=list(cat_ids))
        gt_anns    = [a for a in coco_gt.loadAnns(gt_ann_ids) if a.get("iscrowd", 0) == 0]

        preds_a = preds_a_by_img.get(img_id, [])
        preds_b = preds_b_by_img.get(img_id, [])

        tp_gt_a, tp_pred_a, _ = match(gt_anns, preds_a, args.iou_threshold)
        tp_gt_b, tp_pred_b, _ = match(gt_anns, preds_b, args.iou_threshold)

        fp_a = len(preds_a) - len(tp_pred_a)
        fp_b = len(preds_b) - len(tp_pred_b)

        score = interest_score(tp_gt_a, tp_gt_b, fp_a, fp_b, len(gt_anns))
        scores.append((score, img_id, gt_anns, preds_a, tp_gt_a, tp_pred_a,
                       preds_b, tp_gt_b, tp_pred_b, img_info))

    scores.sort(key=lambda x: -x[0])

    print(f"\nTop {args.top} most differentially interesting frames:")
    for i, (score, img_id, gt_anns, preds_a, tp_gt_a, tp_pred_a,
            preds_b, tp_gt_b, tp_pred_b, img_info) in enumerate(scores[:args.top]):

        tp_a  = len(tp_gt_a);  fp_a = len(preds_a) - tp_a;  fn_a = len(gt_anns) - tp_a
        tp_b  = len(tp_gt_b);  fp_b = len(preds_b) - tp_b;  fn_b = len(gt_anns) - tp_b
        seq   = img_id // 100000
        frame = img_id % 100000
        print(f"  seq{seq:04d}_frame{frame:06d}  score={score:.1f}  "
              f"{args.label_a}: TP={tp_a} FP={fp_a} FN={fn_a}  |  "
              f"{args.label_b}: TP={tp_b} FP={fp_b} FN={fn_b}  |  GT={len(gt_anns)}")

        orig_img = Image.open(
            os.path.join(args.dataset_root, img_info["file_name"])
        ).convert("RGB")

        stats = dict(tp_a=tp_a, fp_a=fp_a, fn_a=fn_a,
                     tp_b=tp_b, fp_b=fp_b, fn_b=fn_b, n_gt=len(gt_anns))

        canvas = build_comparison(
            orig_img, img_info, gt_anns,
            preds_a, tp_gt_a, tp_pred_a,
            preds_b, tp_gt_b, tp_pred_b,
            args.label_a, args.label_b, stats,
        )
        fname = f"rank{i+1:02d}_seq{seq:04d}_frame{frame:06d}.png"
        canvas.save(os.path.join(args.output_dir, fname))

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
