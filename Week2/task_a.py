import os
import sys
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

TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS   = [2, 6, 7, 8, 10, 13, 14, 16, 18]

VALID_CAT_IDS = {1, 3}
CAT_NAMES = {1: "Pedestrian", 3: "Car"}

MASK_COLORS = {1: (0, 255, 0), 3: (0, 0, 255)}


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


def build_coco_gt_segm(original_coco_path: str) -> dict:
    with open(original_coco_path) as f:
        data = json.load(f)

    img_size = {img["id"]: (img["height"], img["width"]) for img in data["images"]}

    for ann in data["annotations"]:
        segm = ann["segmentation"]
        if not isinstance(segm, dict):
            h, w = img_size[ann["image_id"]]
            ann["segmentation"] = {"counts": segm, "size": [h, w]}

    return data


def overlay_masks(image: Image.Image, masks, cat_ids, alpha=0.45) -> Image.Image:
    img_arr = np.array(image).copy()
    for mask, cat_id in zip(masks, cat_ids):
        color = MASK_COLORS.get(cat_id, (255, 0, 0))
        for c, v in enumerate(color):
            img_arr[:, :, c] = np.where(mask, img_arr[:, :, c] * (1 - alpha) + v * alpha,
                                         img_arr[:, :, c])
    return Image.fromarray(img_arr.astype(np.uint8))



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/sam-vit-base")
    p.add_argument("--dataset_root",
                   default="/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/hhome/priubrogent/mcvpol/C5/Week1/R-CNN/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/hhome/priubrogent/mcvpol/C5/Week2/outputs/task_a_w_ignore")
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--qual_images", type=int, default=10)
    p.add_argument("--score_thresh", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    qual_dir = os.path.join(args.output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    print("Loading SAM model")
    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device)
    model.eval()
    print("Model loaded.")


    print("Loading GT annotations …")
    gt_data = build_coco_gt_segm(args.ann_file)
    fixed_gt_path = os.path.join(args.output_dir, "gt_fixed.json")
    with open(fixed_gt_path, "w") as f:
        json.dump(gt_data, f)
    coco_gt = COCO(fixed_gt_path)

    if args.split == "val":
        target_seqs = VAL_SEQS
    elif args.split == "train":
        target_seqs = TRAIN_SEQS
    else:
        target_seqs = TRAIN_SEQS + VAL_SEQS

    all_img_ids = coco_gt.getImgIds()
    img_ids = [iid for iid in all_img_ids if (iid // 100000) in target_seqs]

    img_ids_filtered = []
    for iid in img_ids:
        info = coco_gt.loadImgs([iid])[0]
        p = os.path.join(args.dataset_root, info["file_name"])
        if os.path.exists(p):
            img_ids_filtered.append(iid)

    if args.max_images:
        img_ids_filtered = img_ids_filtered[:args.max_images]

    print(f"Images to process: {len(img_ids_filtered)} "
          f"(split={args.split}, seqs on disk)")

    results = []  
    qual_count = 0

    for img_id in tqdm(img_ids_filtered, desc="SAM inference"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(args.dataset_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        anns = [a for a in anns if a["category_id"] in VALID_CAT_IDS and a.get("iscrowd", 0) == 0]

        if not anns:
            continue

        # Build point prompts: one center point per GT annotation
        input_points = []   # [N, 1, 2]  (one point per object)
        input_labels = []   # [N, 1]

        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = x + w / 2
            cy = y + h / 2
            input_points.append([[cx, cy]])
            input_labels.append([1])   # 1 = foreground

        inputs = processor(
            images=image,
            input_points=[input_points],   # [1, N, 1, 2]
            input_labels=[input_labels],   # [1, N, 1]
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process: get masks at original resolution
        # pred_masks: [1, N, 3, H, W]  (3 candidate masks per object)
        # iou_scores: [1, N, 3]
        pred_masks_list = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        iou_scores = outputs.iou_scores.cpu().numpy()[0]  # [N, 3]

        pred_masks_img = pred_masks_list[0]  # [N, 3, H, W]

        # Collect best mask per annotation
        pred_masks_for_viz = []
        cat_ids_for_viz = []

        for k, ann in enumerate(anns):
            scores_k = iou_scores[k]           # [3]
            best_idx = int(scores_k.argmax())
            best_score = float(scores_k[best_idx])

            if best_score < args.score_thresh:
                continue

            mask_np = pred_masks_img[k, best_idx].numpy().astype(np.uint8)  # (H, W)

            rle = mask_to_rle(mask_np)

            results.append({
                "image_id":    img_id,
                "category_id": ann["category_id"],
                "segmentation": rle,
                "score":        best_score,
            })

            pred_masks_for_viz.append(mask_np)
            cat_ids_for_viz.append(ann["category_id"])

        # ---- Qualitative visualisation ----
        if qual_count < args.qual_images:
            gt_masks_for_viz = [decode_gt_mask(a, img_info) for a in anns]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(image)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            gt_vis = overlay_masks(image.copy(), gt_masks_for_viz,
                                   [a["category_id"] for a in anns])
            # Draw center points
            for ann in anns:
                x, y, w, h = ann["bbox"]
                color = "lime" if ann["category_id"] == 1 else "cyan"
                axes[1].plot(x + w / 2, y + h / 2, "o", color=color, markersize=8)
            axes[1].imshow(gt_vis)
            axes[1].set_title("GT masks + point prompts")
            axes[1].axis("off")

            pred_vis = overlay_masks(image.copy(), pred_masks_for_viz, cat_ids_for_viz)
            axes[2].imshow(pred_vis)
            axes[2].set_title("SAM predictions (point prompts)")
            axes[2].axis("off")

            legend = [
                mpatches.Patch(color=(0, 1, 0), label="Pedestrian"),
                mpatches.Patch(color=(0, 0, 1), label="Car"),
            ]
            fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
            plt.tight_layout()
            seq_id = img_id // 100000
            frame_id = img_id % 100000
            save_path = os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png")
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            plt.close()
            qual_count += 1

    # -----------------------------------------------------------------------
    print(f"\nTotal predictions: {len(results)}")

    if not results:
        print("No predictions – check dataset path and split.")
        return

    # Save predictions
    pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)
    print(f"Predictions saved to {pred_path}")

    # Run COCO eval (segmentation)
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")

    processed_img_ids = list({r["image_id"] for r in results})
    coco_eval.params.imgIds  = processed_img_ids
    coco_eval.params.catIds  = list(VALID_CAT_IDS)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ---- Per-class breakdown ----
    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75",
        "mAP_small", "mAP_medium", "mAP_large",
        "mAR_1", "mAR_10", "mAR_100",
        "mAR_small", "mAR_medium", "mAR_large",
    ]
    metrics = {n: round(float(v), 4)
               for n, v in zip(metric_names, coco_eval.stats)}

    precision = coco_eval.eval["precision"]   # [T, R, K, A, M]
    recall    = coco_eval.eval["recall"]      # [T, K, A, M]
    cat_ids_sorted = sorted(VALID_CAT_IDS)

    for k, cat_id in enumerate(cat_ids_sorted):
        name = CAT_NAMES[cat_id]
        p = precision[:, :, k, 0, 2]   # all sizes, maxDets=100
        r = recall[:, k, 0, 2]
        metrics[f"mAP_{name}"] = round(float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4)
        metrics[f"mAR_{name}"] = round(float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4)

    metrics["model"]        = args.model
    metrics["split"]        = args.split
    metrics["num_images"]   = len(processed_img_ids)
    metrics["num_preds"]    = len(results)
    metrics["prompt_type"]  = "point (GT bbox center)"

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k.startswith("mAP") or k.startswith("mAR")}, indent=2))


if __name__ == "__main__":
    main()
