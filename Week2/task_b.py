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
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from tqdm import tqdm

TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS   = [2, 6, 7, 8, 10, 13, 14, 16, 18]

VALID_CAT_IDS = {1, 3}
CAT_NAMES  = {1: "Pedestrian", 3: "Car"}
MASK_COLORS = {1: (0, 255, 0), 3: (0, 0, 255)}

LABEL_TO_CAT = {
    "person":      1,
    "pedestrian":  1,
    "car":         3,
    "vehicle":     3,
    "automobile":  3,
}


def label_to_cat_id(label: str) -> int | None:
    label_lower = label.lower().rstrip(".")
    for key, cat_id in LABEL_TO_CAT.items():
        if key in label_lower:
            return cat_id
    return None


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
            img_arr[:, :, c] = np.where(
                mask,
                img_arr[:, :, c] * (1 - alpha) + v * alpha,
                img_arr[:, :, c],
            )
    return Image.fromarray(img_arr.astype(np.uint8))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector_id",  default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--segmenter_id", default="facebook/sam-vit-base")
    p.add_argument("--labels",       nargs="+", default=["car", "person"])
    p.add_argument("--threshold",    type=float, default=0.3)

    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week2/outputs/task_b")
    
    p.add_argument("--split",        choices=["train", "val", "all"], default="val")
    p.add_argument("--max_images",   type=int, default=None)
    p.add_argument("--qual_images",  type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    label_tag = "_".join(sorted(args.labels))
    suffix = f"{label_tag}_thr{args.threshold:.2f}"
    output_dir = os.path.join(args.output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Detector: {args.detector_id}  |  Segmenter: {args.segmenter_id}")
    print(f"Labels: {args.labels}  |  threshold: {args.threshold}")

    print("Loading Grounding DINO...")
    detector = pipeline(
        model=args.detector_id,
        task="zero-shot-object-detection",
        device=device,
    )

    print("Loading SAM...")
    sam_processor = AutoProcessor.from_pretrained(args.segmenter_id)
    sam_model     = AutoModelForMaskGeneration.from_pretrained(args.segmenter_id).to(device)
    sam_model.eval()

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
    img_ids = [
        iid for iid in img_ids
        if os.path.exists(os.path.join(args.dataset_root, coco_gt.loadImgs([iid])[0]["file_name"]))
    ]
    if args.max_images:
        img_ids = img_ids[:args.max_images]

    print(f"Images to process: {len(img_ids)} (split={args.split})")

    dino_labels = [l if l.endswith(".") else l + "." for l in args.labels]

    results    = []
    qual_count = 0

    for img_id in tqdm(img_ids, desc="Grounded SAM inference"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(args.dataset_root, img_info["file_name"])
        image    = Image.open(img_path).convert("RGB")

        detections = detector(image, candidate_labels=dino_labels, threshold=args.threshold)
        if not detections:
            continue

        valid_dets = []
        for det in detections:
            cat_id = label_to_cat_id(det["label"])
            if cat_id is not None:
                valid_dets.append((det, cat_id))
        if not valid_dets:
            continue

        boxes = [
            [d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]]
            for d, _ in valid_dets
        ]
        inputs = sam_processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = sam_model(**inputs)

        masks_list = sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )
        iou_scores    = outputs.iou_scores.cpu().numpy()[0]
        pred_masks_np = masks_list[0].cpu().float()

        pred_masks_viz = []
        cat_ids_viz    = []

        for k, (det, cat_id) in enumerate(valid_dets):
            best_idx = int(iou_scores[k].argmax())
            mask_np  = (pred_masks_np[k, best_idx].numpy() > 0).astype(np.uint8)
            results.append({
                "image_id":     img_id,
                "category_id":  cat_id,
                "segmentation": mask_to_rle(mask_np),
                "score":        det["score"],
            })
            pred_masks_viz.append(mask_np)
            cat_ids_viz.append(cat_id)

        if qual_count < args.qual_images:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].imshow(image)
            for det, cat_id in valid_dets:
                b     = det["box"]
                color = "lime" if cat_id == 1 else "cyan"
                rect  = plt.Rectangle(
                    (b["xmin"], b["ymin"]),
                    b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                    linewidth=2, edgecolor=color, facecolor="none",
                )
                axes[0].add_patch(rect)
                axes[0].text(b["xmin"], b["ymin"] - 4,
                             f'{det["label"]} {det["score"]:.2f}',
                             color=color, fontsize=7, fontweight="bold")
            axes[0].set_title(f"Grounding DINO (thr={args.threshold})")
            axes[0].axis("off")

            pred_vis = overlay_masks(image.copy(), pred_masks_viz, cat_ids_viz)
            axes[1].imshow(pred_vis)
            axes[1].set_title("SAM predictions")
            axes[1].axis("off")

            legend = [
                mpatches.Patch(color=(0, 1, 0), label="Pedestrian"),
                mpatches.Patch(color=(0, 0, 1), label="Car"),
            ]
            fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
            plt.tight_layout()
            seq_id   = img_id // 100000
            frame_id = img_id % 100000
            plt.savefig(
                os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png"),
                dpi=100, bbox_inches="tight",
            )
            plt.close()
            qual_count += 1

    print(f"\nTotal predictions: {len(results)}")
    if not results:
        print("No predictions – check dataset path, labels, and threshold.")
        return

    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)

    coco_dt   = coco_gt.loadRes(results)
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
    metrics = {n: round(float(v), 4) for n, v in zip(metric_names, coco_eval.stats)}

    precision = coco_eval.eval["precision"]
    recall    = coco_eval.eval["recall"]

    for k, cat_id in enumerate(sorted(VALID_CAT_IDS)):
        name = CAT_NAMES[cat_id]
        p = precision[:, :, k, 0, 2]
        r = recall[:, k, 0, 2]
        metrics[f"mAP_{name}"] = round(float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4)
        metrics[f"mAR_{name}"] = round(float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4)

    metrics.update({
        "detector":   args.detector_id,
        "segmenter":  args.segmenter_id,
        "labels":     args.labels,
        "threshold":  args.threshold,
        "split":      args.split,
        "num_images": len(processed_img_ids),
        "num_preds":  len(results),
    })

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k.startswith("mAP") or k.startswith("mAR")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
