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
from transformers import AutoModelForMaskGeneration, AutoProcessor
from ultralytics import YOLO
from tqdm import tqdm

# The LoRA checkpoint was saved with Week1.utils.utils.LoRAConv2d in the pickle,
# so Week1 must be importable before torch.load is called internally by YOLO.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from Week1.utils.utils import LoRAConv2d  # noqa: F401 — needed for pickle deserialization

TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS   = [2, 6, 7, 8, 10, 13, 14, 16, 18]

VALID_CAT_IDS = {1, 3}
CAT_NAMES     = {1: "Pedestrian", 3: "Car"}
MASK_COLORS   = {1: (0, 255, 0), 3: (0, 0, 255)}

# yolo11x (COCO-pretrained): person=0, car=2
ORIGINAL_CLASS_MAP  = {0: 1, 2: 3}
# LoRA-finetuned on KITTI-MOTS: Pedestrian=0, Car=1
FINETUNED_CLASS_MAP = {0: 1, 1: 3}


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
    p.add_argument("--finetuned", action="store_true",
                   help="Use the LoRA-finetuned YOLO instead of the pretrained one")
    p.add_argument("--original_weights", default="yolo11x.pt")
    p.add_argument("--finetuned_weights",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/YOLO/yolo_lora_best.pt")
    p.add_argument("--segmenter_id", default="facebook/sam-vit-base")
    p.add_argument("--threshold",    type=float, default=0.25,
                   help="YOLO confidence threshold (ultralytics default: 0.25)")
    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week2/outputs/task_c")
    p.add_argument("--split",       choices=["train", "val", "all"], default="val")
    p.add_argument("--max_images",  type=int, default=None)
    p.add_argument("--qual_per_seq", type=int, default=5,
                   help="Qualitative frames to save per sequence")
    return p.parse_args()


def run_coco_eval(coco_gt, results, iou_type, processed_img_ids):
    coco_dt   = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = processed_img_ids
    coco_eval.params.catIds = list(VALID_CAT_IDS)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def extract_metrics(coco_eval, prefix):
    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75",
        "mAP_small", "mAP_medium", "mAP_large",
        "mAR_1", "mAR_10", "mAR_100",
        "mAR_small", "mAR_medium", "mAR_large",
    ]
    metrics = {f"{prefix}/{n}": round(float(v), 4)
               for n, v in zip(metric_names, coco_eval.stats)}

    precision = coco_eval.eval["precision"]
    recall    = coco_eval.eval["recall"]
    for k, cat_id in enumerate(sorted(VALID_CAT_IDS)):
        name = CAT_NAMES[cat_id]
        p = precision[:, :, k, 0, 2]
        r = recall[:, k, 0, 2]
        metrics[f"{prefix}/mAP_{name}"] = round(float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4)
        metrics[f"{prefix}/mAR_{name}"] = round(float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4)

    return metrics


def main():
    args = parse_args()

    model_tag    = "finetuned" if args.finetuned else "original"
    output_dir   = os.path.join(args.output_dir, f"{model_tag}_thr{args.threshold:.2f}")
    qual_dir     = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights    = args.finetuned_weights if args.finetuned else args.original_weights
    class_map  = FINETUNED_CLASS_MAP   if args.finetuned else ORIGINAL_CLASS_MAP
    print(f"Device: {device}  |  YOLO: {weights}  |  Segmenter: {args.segmenter_id}")
    print(f"Mode: {model_tag}  |  threshold: {args.threshold}")

    print("Loading YOLO...")
    detector = YOLO(weights)
    detector.to(device)
    if args.finetuned:
        detector.model.fuse = lambda *args, **kwargs: detector.model

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

    segm_results = []
    bbox_results = []
    qual_counts  = {}

    for img_id in tqdm(img_ids, desc="YOLO + SAM inference"):
        img_info = coco_gt.loadImgs([img_id])[0]
        img_path = os.path.join(args.dataset_root, img_info["file_name"])
        image    = Image.open(img_path).convert("RGB")

        yolo_results = detector(image, conf=args.threshold, verbose=False)[0]

        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            continue

        raw_classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
        raw_boxes   = yolo_results.boxes.xyxy.cpu().numpy()
        raw_scores  = yolo_results.boxes.conf.cpu().numpy()

        valid_dets = []
        for cls, box, score in zip(raw_classes, raw_boxes, raw_scores):
            cat_id = class_map.get(int(cls))
            if cat_id is not None:
                valid_dets.append((box, score, cat_id))

        if not valid_dets:
            continue

        boxes = [[float(x) for x in det[0]] for det in valid_dets]

        for box, score, cat_id in valid_dets:
            x1, y1, x2, y2 = box.tolist()
            bbox_results.append({
                "image_id":    img_id,
                "category_id": cat_id,
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       float(score),
            })

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

        for k, (box, score, cat_id) in enumerate(valid_dets):
            best_idx = int(iou_scores[k].argmax())
            mask_np  = (pred_masks_np[k, best_idx].numpy() > 0).astype(np.uint8)
            segm_results.append({
                "image_id":     img_id,
                "category_id":  cat_id,
                "segmentation": mask_to_rle(mask_np),
                "score":        float(score),
            })
            pred_masks_viz.append(mask_np)
            cat_ids_viz.append(cat_id)

        seq_id   = img_id // 100000
        frame_id = img_id % 100000
        if qual_counts.get(seq_id, 0) < args.qual_per_seq:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].imshow(image)
            for box, score, cat_id in valid_dets:
                x1, y1, x2, y2 = box
                color = "lime" if cat_id == 1 else "cyan"
                rect  = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor="none",
                )
                axes[0].add_patch(rect)
                axes[0].text(x1, y1 - 4,
                             f'{CAT_NAMES[cat_id]} {score:.2f}',
                             color=color, fontsize=7, fontweight="bold")
            axes[0].set_title(f"YOLO {model_tag} (conf={args.threshold})")
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
            plt.savefig(
                os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png"),
                dpi=100, bbox_inches="tight",
            )
            plt.close()
            qual_counts[seq_id] = qual_counts.get(seq_id, 0) + 1

    print(f"\nTotal predictions: {len(segm_results)}")
    if not segm_results:
        print("No predictions — check dataset path, weights, and threshold.")
        return

    with open(os.path.join(output_dir, "predictions_segm.json"), "w") as f:
        json.dump(segm_results, f)
    with open(os.path.join(output_dir, "predictions_bbox.json"), "w") as f:
        json.dump(bbox_results, f)

    processed_img_ids = list({r["image_id"] for r in segm_results})

    print("\n--- Segmentation metrics ---")
    segm_eval = run_coco_eval(coco_gt, segm_results, "segm", processed_img_ids)

    print("\n--- Detection (bbox) metrics ---")
    bbox_eval = run_coco_eval(coco_gt, bbox_results, "bbox", processed_img_ids)

    metrics = {}
    metrics.update(extract_metrics(segm_eval, "segm"))
    metrics.update(extract_metrics(bbox_eval, "bbox"))
    metrics.update({
        "model":      weights,
        "mode":       model_tag,
        "segmenter":  args.segmenter_id,
        "threshold":  args.threshold,
        "split":      args.split,
        "num_images": len(processed_img_ids),
        "num_preds":  len(segm_results),
    })

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    print(json.dumps(
        {k: v for k, v in metrics.items() if k.startswith("segm/mAP") or k.startswith("bbox/mAP")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
