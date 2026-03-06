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


def get_point_prompts(ann, img_info, strategy, num_points, rng):
    """Return (points, labels) for a single annotation.

    points : list of [x, y]  (pixel coordinates)
    labels : list of int     (1 = foreground)
    """
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

    raise ValueError(f"Unknown strategy: {strategy}")


def get_neg_points(all_anns, img_info, n_neg, rng):
    """Sample n_neg background points that fall outside every GT bbox."""
    H, W = img_info["height"], img_info["width"]
    bboxes = [a["bbox"] for a in all_anns]
    neg_pts = []
    for _ in range(n_neg * 20):          # max attempts
        if len(neg_pts) == n_neg:
            break
        px = float(rng.uniform(0, W))
        py = float(rng.uniform(0, H))
        if not any(bx <= px <= bx + bw and by <= py <= by + bh
                   for bx, by, bw, bh in bboxes):
            neg_pts.append([px, py])
    return neg_pts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/sam-vit-base")
    p.add_argument("--dataset_root",
                   default="/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/hhome/priubrogent/mcvpol/C5/Week1/R-CNN/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/hhome/priubrogent/mcvpol/C5/Week2/outputs/task_a_experiments")
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--qual_images", type=int, default=10)
    p.add_argument("--score_thresh", type=float, default=0.0)
    p.add_argument("--strategy",
                   choices=["bbox_center", "mask_centroid", "random_mask", "random_bbox"],
                   default="bbox_center")
    p.add_argument("--num_points", type=int, default=1)
    p.add_argument("--neg_points", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-suffix output dir so different experiments don't overwrite each other
    suffix = args.strategy
    if args.num_points > 1:
        suffix += f"_n{args.num_points}"
    if args.neg_points > 0:
        suffix += f"_neg{args.neg_points}"
    output_dir = os.path.join(args.output_dir, suffix)

    os.makedirs(output_dir, exist_ok=True)
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")
    print(f"Strategy: {args.strategy} | num_points: {args.num_points} | neg_points: {args.neg_points}")

    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device)
    model.eval()

    print("Loading GT annotations...")
    gt_data = build_coco_gt_segm(args.ann_file)
    fixed_gt_path = os.path.join(output_dir, "gt_fixed.json")
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
    img_ids_filtered = [
        iid for iid in img_ids
        if os.path.exists(os.path.join(args.dataset_root,
                                       coco_gt.loadImgs([iid])[0]["file_name"]))
    ]

    if args.max_images:
        img_ids_filtered = img_ids_filtered[:args.max_images]

    print(f"Images to process: {len(img_ids_filtered)} (split={args.split})")

    results = []
    qual_count = 0

    for img_id in tqdm(img_ids_filtered, desc="SAM inference"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(args.dataset_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        anns = [a for a in anns if a["category_id"] in VALID_CAT_IDS and a.get("iscrowd", 0) == 0]

        if not anns:
            continue

        # Sample negative background points once per image (shared across objects)
        neg_pts = get_neg_points(anns, img_info, args.neg_points, rng) if args.neg_points > 0 else []

        # Build per-object point prompts
        input_points = []
        input_labels = []
        pts_per_ann = []  # keep for visualisation

        for ann in anns:
            pts, labs = get_point_prompts(ann, img_info, args.strategy, args.num_points, rng)
            pts_per_ann.append((pts, ann["category_id"]))
            pts  = pts  + neg_pts
            labs = labs + [0] * len(neg_pts)
            input_points.append(pts)
            input_labels.append(labs)

        inputs = processor(
            images=image,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        pred_masks_list = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        iou_scores = outputs.iou_scores.cpu().numpy()[0]  # [N, 3]
        pred_masks_img = pred_masks_list[0]               # [N, 3, H, W]

        pred_masks_for_viz = []
        cat_ids_for_viz = []

        for k, ann in enumerate(anns):
            scores_k = iou_scores[k]
            best_idx = int(scores_k.argmax())
            best_score = float(scores_k[best_idx])

            if best_score < args.score_thresh:
                continue

            mask_np = pred_masks_img[k, best_idx].numpy().astype(np.uint8)
            results.append({
                "image_id":     img_id,
                "category_id":  ann["category_id"],
                "segmentation": mask_to_rle(mask_np),
                "score":        best_score,
            })
            pred_masks_for_viz.append(mask_np)
            cat_ids_for_viz.append(ann["category_id"])

        # Qualitative visualisation
        if qual_count < args.qual_images:
            gt_masks_for_viz = [decode_gt_mask(a, img_info) for a in anns]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(image)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            gt_vis = overlay_masks(image.copy(), gt_masks_for_viz,
                                   [a["category_id"] for a in anns])
            axes[1].imshow(gt_vis)
            for pts, cat_id in pts_per_ann:
                color = "lime" if cat_id == 1 else "cyan"
                for px, py in pts:
                    axes[1].plot(px, py, "o", color=color, markersize=6)
            axes[1].set_title(f"GT masks + {args.strategy} prompts")
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
            seq_id = img_id // 100000
            frame_id = img_id % 100000
            plt.savefig(os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close()
            qual_count += 1

    print(f"\nTotal predictions: {len(results)}")
    if not results:
        print("No predictions – check dataset path and split.")
        return

    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)
    print(f"Predictions saved to {pred_path}")

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")

    processed_img_ids = list({r["image_id"] for r in results})
    coco_eval.params.imgIds = processed_img_ids
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

    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    recall    = coco_eval.eval["recall"]     # [T, K, A, M]

    for k, cat_id in enumerate(sorted(VALID_CAT_IDS)):
        name = CAT_NAMES[cat_id]
        p = precision[:, :, k, 0, 2]
        r = recall[:, k, 0, 2]
        metrics[f"mAP_{name}"] = round(float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4)
        metrics[f"mAR_{name}"] = round(float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4)

    metrics.update({
        "model":       args.model,
        "split":       args.split,
        "strategy":    args.strategy,
        "num_points":  args.num_points,
        "neg_points":  args.neg_points,
        "num_images":  len(processed_img_ids),
        "num_preds":   len(results),
        "prompt_type": f"{args.strategy} (n={args.num_points}, neg={args.neg_points})",
    })

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k.startswith("mAP") or k.startswith("mAR")}, indent=2))


if __name__ == "__main__":
    main()
