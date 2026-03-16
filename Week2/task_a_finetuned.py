"""
Evaluate a (fine-tuned) SAM model with all task_a prompt strategies + GT bbox prompts.

Mirrors run_task_a.sh experiments exactly, adds gt_bbox (input_boxes), and supports
loading a PEFT adapter or merged .pt checkpoint from task_e for apples-to-apples
comparison with the pretrained baseline.

Experiments run (matching run_task_a.sh + gt_bbox):
    bbox_center     n=1
    mask_centroid   n=1
    random_mask     n=1, 3, 5
    random_bbox     n=1, 3, 5
    sift_best       n=1
    sift_topk       n=1, 3, 5
    gt_bbox         (GT bounding-box prompt, like task_e_inference.py)

Examples
--------
# Pretrained baseline (no checkpoint):
python task_a_finetuned.py

# Fine-tuned PEFT adapter:
python task_a_finetuned.py \
    --peft_adapter outputs/task_e_lora/lora_r8_focal_dice_noaug_sam-vit-base/best_adapter

# Merged checkpoint:
python task_a_finetuned.py \
    --checkpoint outputs/task_e_lora/lora_r8_focal_dice_noaug_sam-vit-base/best_model_merged.pt

# Subset of experiments only:
python task_a_finetuned.py \
    --peft_adapter ... \
    --experiments bbox_center gt_bbox random_mask_n3
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from transformers import SamModel, SamProcessor
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS   = [2, 6, 7, 8, 10, 13, 14, 16, 18]

VALID_CAT_IDS = {1, 3}
CAT_NAMES     = {1: "Pedestrian", 3: "Car"}
MASK_COLORS   = {1: (0, 255, 0), 3: (0, 0, 255)}

# Ordered list of (experiment_key, strategy, num_points) matching run_task_a.sh + gt_bbox
EXPERIMENTS = [
    ("bbox_center",    "bbox_center",    1),
    ("mask_centroid",  "mask_centroid",  1),
    ("random_mask_n1", "random_mask",    1),
    ("random_mask_n3", "random_mask",    3),
    ("random_mask_n5", "random_mask",    5),
    ("random_bbox_n1", "random_bbox",    1),
    ("random_bbox_n3", "random_bbox",    3),
    ("random_bbox_n5", "random_bbox",    5),
    ("sift_best",      "sift_best",      1),
    ("sift_topk_n1",   "sift_topk",      1),
    ("sift_topk_n3",   "sift_topk",      3),
    ("sift_topk_n5",   "sift_topk",      5),
    ("gt_bbox",        "gt_bbox",        1),   # uses input_boxes, num_points ignored
]
ALL_EXP_KEYS = [e[0] for e in EXPERIMENTS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_gt_mask(ann, img_info):
    segm = ann["segmentation"]
    h, w = img_info["height"], img_info["width"]
    if isinstance(segm, dict):
        rle = segm
    else:
        rle = {"counts": segm, "size": [h, w]}
    return mask_util.decode(rle).astype(np.uint8)


def mask_to_rle(mask: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_coco_gt_segm(original_coco_path: str, out_dir: str):
    with open(original_coco_path) as f:
        data = json.load(f)
    img_size = {img["id"]: (img["height"], img["width"]) for img in data["images"]}
    for ann in data["annotations"]:
        segm = ann["segmentation"]
        if not isinstance(segm, dict):
            h, w = img_size[ann["image_id"]]
            ann["segmentation"] = {"counts": segm, "size": [h, w]}
    fixed_path = os.path.join(out_dir, "gt_fixed.json")
    with open(fixed_path, "w") as f:
        json.dump(data, f)
    return COCO(fixed_path)


def overlay_masks(image: Image.Image, masks, cat_ids, alpha=0.45) -> Image.Image:
    img_arr = np.array(image).copy()
    for mask, cat_id in zip(masks, cat_ids):
        color = MASK_COLORS.get(cat_id, (255, 0, 0))
        for c, v in enumerate(color):
            img_arr[:, :, c] = np.where(mask,
                                         img_arr[:, :, c] * (1 - alpha) + v * alpha,
                                         img_arr[:, :, c])
    return Image.fromarray(img_arr.astype(np.uint8))


def get_point_prompts(ann, img_info, strategy, num_points, rng):
    """Return (points, labels) for a single annotation (point-based strategies)."""
    x, y, w, h = ann["bbox"]

    if strategy == "bbox_center":
        return [[x + w / 2, y + h / 2]], [1]

    if strategy == "mask_centroid":
        mask = decode_gt_mask(ann, img_info)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [[x + w / 2, y + h / 2]], [1]
        return [[float(np.mean(xs)), float(np.mean(ys))]], [1]

    if strategy == "random_mask":
        mask = decode_gt_mask(ann, img_info)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            pts = [[x + w / 2, y + h / 2]] * num_points
        else:
            idxs = rng.choice(len(xs), size=min(num_points, len(xs)), replace=False)
            pts = [[float(xs[i]), float(ys[i])] for i in idxs]
            while len(pts) < num_points:
                pts.append(pts[-1])
        return pts, [1] * len(pts)

    if strategy == "random_bbox":
        pts = [[float(rng.uniform(x, x + w)), float(rng.uniform(y, y + h))]
               for _ in range(num_points)]
        return pts, [1] * len(pts)

    if strategy in ("sift_best", "sift_topk"):
        import cv2
        img_np = np.array(img_info.get("_pil_image"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create()
        kps  = sift.detect(gray, None)
        mask = decode_gt_mask(ann, img_info)
        H, W = mask.shape
        inside = [
            (kp.pt[0], kp.pt[1], kp.response) for kp in kps
            if 0 <= int(kp.pt[1]) < H and 0 <= int(kp.pt[0]) < W
            and mask[int(kp.pt[1]), int(kp.pt[0])] > 0
        ]
        if not inside:
            return [[x + w / 2, y + h / 2]], [1]
        if strategy == "sift_best":
            best = max(inside, key=lambda t: t[2])
            return [[best[0], best[1]]], [1]
        inside.sort(key=lambda t: t[2], reverse=True)
        pts = [[t[0], t[1]] for t in inside[:num_points]]
        while len(pts) < num_points:
            pts.append(pts[-1])
        return pts, [1] * len(pts)

    raise ValueError(f"Unknown strategy: {strategy}")


def get_neg_points(all_anns, img_info, n_neg, rng):
    H, W = img_info["height"], img_info["width"]
    bboxes = [a["bbox"] for a in all_anns]
    neg_pts = []
    for _ in range(n_neg * 20):
        if len(neg_pts) == n_neg:
            break
        px = float(rng.uniform(0, W))
        py = float(rng.uniform(0, H))
        if not any(bx <= px <= bx + bw and by <= py <= by + bh
                   for bx, by, bw, bh in bboxes):
            neg_pts.append([px, py])
    return neg_pts


# ---------------------------------------------------------------------------
# Per-experiment evaluation
# ---------------------------------------------------------------------------

def run_experiment(model, processor, device, coco_gt, img_ids, dataset_root,
                   exp_key, strategy, num_points, neg_points, score_thresh,
                   qual_per_seq, output_dir, rng, model_label):

    out_dir  = os.path.join(output_dir, exp_key)
    qual_dir = os.path.join(out_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    is_bbox_strategy = (strategy == "gt_bbox")

    results     = []
    qual_counts = {}

    for img_id in tqdm(img_ids, desc=f"[{exp_key}]"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(dataset_root, img_info["file_name"])
        image    = Image.open(img_path).convert("RGB")
        img_info["_pil_image"] = np.array(image)

        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns    = coco_gt.loadAnns(ann_ids)
        anns    = [a for a in anns
                   if a["category_id"] in VALID_CAT_IDS and a.get("iscrowd", 0) == 0
                   and a["bbox"][2] >= 2 and a["bbox"][3] >= 2]
        if not anns:
            continue

        # ---- Build model inputs ----
        if is_bbox_strategy:
            boxes = [[a["bbox"][0], a["bbox"][1],
                      a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                     for a in anns]
            inputs = processor(
                images=image,
                input_boxes=[[boxes]],   # [batch, n_objects, 1, 4]
                return_tensors="pt",
            ).to(device)
            pts_per_ann = None  # no points to visualise
        else:
            neg_pts = (get_neg_points(anns, img_info, neg_points, rng)
                       if neg_points > 0 else [])
            input_points = []
            input_labels = []
            pts_per_ann  = []
            for ann in anns:
                pts, labs = get_point_prompts(ann, img_info, strategy, num_points, rng)
                # Fallbacks may return fewer points than requested; pad to keep homogeneous shape
                while len(pts) < num_points:
                    pts  = pts  + [pts[-1]]
                    labs = labs + [labs[-1]]
                pts_per_ann.append((pts, ann["category_id"]))
                input_points.append(pts  + neg_pts)
                input_labels.append(labs + [0] * len(neg_pts))
            inputs = processor(
                images=image,
                input_points=[input_points],
                input_labels=[input_labels],
                return_tensors="pt",
            ).to(device)

        # ---- Forward pass ----
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=not is_bbox_strategy)

        pred_masks_list = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        pred_masks_img = pred_masks_list[0]   # [N, num_masks, H, W]

        pred_masks_for_viz = []
        cat_ids_for_viz    = []

        for k, ann in enumerate(anns):
            if is_bbox_strategy:
                # Single mask output; threshold at 0
                mask_np = (pred_masks_img[k, 0].numpy() > 0).astype(np.uint8)
                score   = 1.0
            else:
                iou_scores = outputs.iou_scores.cpu().numpy()[0]  # [N, 3]
                scores_k   = iou_scores[k]
                best_idx   = int(scores_k.argmax())
                score      = float(scores_k[best_idx])
                if score < score_thresh:
                    continue
                mask_np = pred_masks_img[k, best_idx].numpy().astype(np.uint8)

            results.append({
                "image_id":     img_id,
                "category_id":  ann["category_id"],
                "segmentation": mask_to_rle(mask_np),
                "score":        score,
            })
            pred_masks_for_viz.append(mask_np)
            cat_ids_for_viz.append(ann["category_id"])

        # ---- Qualitative ----
        seq_id   = img_id // 100000
        frame_id = img_id % 100000
        if qual_counts.get(seq_id, 0) < qual_per_seq:
            gt_masks_for_viz = [decode_gt_mask(a, img_info) for a in anns]
            W_img, H_img = image.size
            dpi = 100
            fig, axes = plt.subplots(1, 3, figsize=(W_img * 3 / dpi, H_img / dpi), dpi=dpi)

            axes[0].imshow(image)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            gt_vis = overlay_masks(image.copy(), gt_masks_for_viz,
                                   [a["category_id"] for a in anns])
            axes[1].imshow(gt_vis)
            if is_bbox_strategy:
                for ann in anns:
                    bx, by, bw, bh = ann["bbox"]
                    color = "lime" if ann["category_id"] == 1 else "cyan"
                    rect  = mpatches.Rectangle((bx, by), bw, bh,
                                               linewidth=2, edgecolor=color, facecolor="none")
                    axes[1].add_patch(rect)
                axes[1].set_title("GT masks + GT bbox prompts")
            else:
                for pts, cat_id in pts_per_ann:
                    color = "lime" if cat_id == 1 else "cyan"
                    for px, py in pts:
                        axes[1].plot(px, py, "o", color=color, markersize=6)
                axes[1].set_title(f"GT masks + {exp_key} prompts")
            axes[1].axis("off")

            pred_vis = overlay_masks(image.copy(), pred_masks_for_viz, cat_ids_for_viz)
            axes[2].imshow(pred_vis)
            axes[2].set_title("SAM predictions")
            axes[2].axis("off")

            legend = [
                mpatches.Patch(color=(0, 1, 0), label="Pedestrian"),
                mpatches.Patch(color=(0, 0, 1), label="Car"),
            ]
            fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png"),
                        dpi=dpi, bbox_inches="tight")
            plt.close()
            qual_counts[seq_id] = qual_counts.get(seq_id, 0) + 1

    print(f"  [{exp_key}] Total predictions: {len(results)}")
    if not results:
        print(f"  [{exp_key}] WARNING: no predictions – skipping metrics.")
        return {}

    with open(os.path.join(out_dir, "predictions.json"), "w") as f:
        json.dump(results, f)

    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.imgIds = list({r["image_id"] for r in results})
    coco_eval.params.catIds = list(VALID_CAT_IDS)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75",
        "mAP_small", "mAP_medium", "mAP_large",
        "mAR_1", "mAR_10", "mAR_100",
        "mAR_small", "mAR_medium", "mAR_large",
    ]
    metrics = {n: round(float(v), 4)
               for n, v in zip(metric_names, coco_eval.stats)}

    precision = coco_eval.eval["precision"]
    recall    = coco_eval.eval["recall"]
    for k, cat_id in enumerate(sorted(VALID_CAT_IDS)):
        name = CAT_NAMES[cat_id]
        p = precision[:, :, k, 0, 2]
        r = recall[:, k, 0, 2]
        metrics[f"mAP_{name}"] = round(
            float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4)
        metrics[f"mAR_{name}"] = round(
            float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4)

    metrics.update({
        "model":      model_label,
        "exp_key":    exp_key,
        "strategy":   strategy,
        "num_points": num_points,
        "neg_points": neg_points,
        "num_images": len({r["image_id"] for r in results}),
        "num_preds":  len(results),
    })

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  [{exp_key}] Metrics saved to {out_dir}/metrics.json")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate (fine-tuned) SAM with all task_a strategies + GT bbox"
    )
    p.add_argument("--model_id", default="facebook/sam-vit-base")
    p.add_argument("--checkpoint", default=None,
                   help="Path to merged fine-tuned .pt state-dict")
    p.add_argument("--peft_adapter", default=None,
                   help="Path to PEFT adapter directory")
    p.add_argument("--experiments", nargs="+", choices=ALL_EXP_KEYS,
                   default=ALL_EXP_KEYS,
                   help="Which experiments to run (default: all). "
                        f"Choices: {ALL_EXP_KEYS}")
    p.add_argument("--neg_points", type=int, default=0)
    p.add_argument("--score_thresh", type=float, default=0.0)
    p.add_argument("--qual_per_seq", type=int, default=999999999)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week2/outputs/task_a_finetuned")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint:
        model_label = "finetuned_" + os.path.splitext(os.path.basename(args.checkpoint))[0]
    elif args.peft_adapter:
        model_label = "peft_" + os.path.basename(args.peft_adapter.rstrip("/"))
    else:
        model_label = "pretrained_" + args.model_id.split("/")[-1]

    output_dir = os.path.join(args.output_dir, model_label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model label  : {model_label}")
    print(f"Device       : {device}")
    print(f"Experiments  : {args.experiments}")

    # Load model once
    print(f"\nLoading base model {args.model_id}...")
    model = SamModel.from_pretrained(args.model_id)

    if args.checkpoint:
        print(f"Loading merged checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    elif args.peft_adapter:
        from peft import PeftModel
        print(f"Loading PEFT adapter: {args.peft_adapter}")
        model = PeftModel.from_pretrained(model, args.peft_adapter)
        model = model.merge_and_unload()

    model = model.to(device)
    model.eval()

    processor = SamProcessor.from_pretrained(args.model_id)

    # Load annotations (write gt_fixed.json once into the run root)
    print("Loading GT annotations...")
    coco_gt = build_coco_gt_segm(args.ann_file, output_dir)

    if args.split == "val":
        target_seqs = VAL_SEQS
    elif args.split == "train":
        target_seqs = TRAIN_SEQS
    else:
        target_seqs = TRAIN_SEQS + VAL_SEQS

    all_img_ids = coco_gt.getImgIds()
    img_ids = [
        iid for iid in all_img_ids
        if (iid // 100000) in target_seqs
        and os.path.exists(os.path.join(args.dataset_root,
                                        coco_gt.loadImgs([iid])[0]["file_name"]))
    ]
    if args.max_images:
        img_ids = img_ids[:args.max_images]

    print(f"Images to process: {len(img_ids)} (split={args.split})\n")

    # Run experiments
    exp_lookup = {e[0]: e for e in EXPERIMENTS}
    all_metrics = {}

    for exp_key in args.experiments:
        _, strategy, num_points = exp_lookup[exp_key]
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_key}  (strategy={strategy}, n={num_points})")
        print('='*60)
        exp_rng = np.random.default_rng(args.seed)
        metrics = run_experiment(
            model, processor, device, coco_gt, img_ids, args.dataset_root,
            exp_key, strategy, num_points, args.neg_points, args.score_thresh,
            args.qual_per_seq, output_dir, exp_rng, model_label,
        )
        all_metrics[exp_key] = metrics

    # Summary table
    print("\n\n" + "="*75)
    print(f"SUMMARY — model: {model_label}  |  split: {args.split}")
    print("="*75)
    hdr = f"{'Experiment':<22} {'mAP@.5:.95':>12} {'mAP@.50':>10} {'mAP@.75':>10} {'mAP_Ped':>10} {'mAP_Car':>10}"
    print(hdr)
    print("-" * len(hdr))
    for exp_key, m in all_metrics.items():
        if not m:
            print(f"{exp_key:<22} {'N/A':>12}")
            continue
        print(f"{exp_key:<22} "
              f"{m.get('mAP_0.50_0.95', float('nan')):>12.4f} "
              f"{m.get('mAP_0.50',      float('nan')):>10.4f} "
              f"{m.get('mAP_0.75',      float('nan')):>10.4f} "
              f"{m.get('mAP_Pedestrian', float('nan')):>10.4f} "
              f"{m.get('mAP_Car',        float('nan')):>10.4f}")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"model_label": model_label, "split": args.split,
                   "experiments": all_metrics}, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
