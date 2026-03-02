"""
Baseline evaluation of the pretrained RT-DETR model (no finetuning) on
the KITTI-MOTS validation split.  Run this before finetuning to get a
baseline mAP to compare against.
"""
import os
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import coco_evaluation, draw_bboxes, filter_results, VAL_SEQS

from pycocotools.coco import COCO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH    = "/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02"
GT_PATH         = os.path.join(os.path.dirname(__file__), "..", "kitti_mots_to_coco_gt.json")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "pretrained_eval")
CHECKPOINT      = "PekingU/rtdetr_r101vd"

# Set to True to also save visualisation images (first frame of each val sequence)
SAVE_VIS        = True
VIS_DIR         = os.path.join(OUTPUT_DIR, "visualizations")


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pretrained model as-is (80 COCO classes, no head replacement)
    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)
    model     = RTDetrForObjectDetection.from_pretrained(CHECKPOINT).to(device)
    model.eval()

    COCO_LABELS = model.config.id2label   # maps label_id → class name string

    # Load ground truth
    if not os.path.exists(GT_PATH):
        print(f"GT file not found: {GT_PATH}\nRun utils/GT_conversor.py first.")
        return
    coco_gt   = COCO(GT_PATH)
    valid_ids = set(coco_gt.getImgIds())

    results_list = []

    print(f"Running inference on {len(VAL_SEQS)} validation sequences...")
    for seq_idx in tqdm(VAL_SEQS, desc="Sequences", position=0):
        folder = Path(DATASET_PATH) / f"{seq_idx:04d}"
        if not folder.exists():
            continue

        img_files = sorted(folder.glob("*.png"))
        for img_path in tqdm(img_files, desc=f"Seq {seq_idx:04d}", position=1, leave=False):
            frame_idx      = int(img_path.stem)
            unique_img_id  = seq_idx * 100000 + frame_idx

            if unique_img_id not in valid_ids:
                continue

            image  = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # (H, W) order as required by post_process_object_detection
            target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)

            # threshold=0 keeps all predictions → required for a correct mAP curve
            preds = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]

            for score, label, bbox in zip(preds["scores"], preds["labels"], preds["boxes"]):
                label_id = label.item()
                # Keep only person (COCO 1) and car (COCO 3)
                if label_id not in {1, 3}:
                    continue
                x1, y1, x2, y2 = bbox.tolist()
                results_list.append({
                    "image_id":   unique_img_id,
                    "category_id": label_id,
                    "bbox":       [x1, y1, x2 - x1, y2 - y1],  # COCO [x, y, w, h]
                    "score":      score.item(),
                })

    # --- COCO Evaluation ---
    if not results_list:
        print("No detections produced — check DATASET_PATH and GT_PATH.")
        return

    print(f"\nTotal detections collected: {len(results_list)}")
    coco_evaluation(
        results_list,
        coco_gt,
        OUTPUT_DIR,
        file_name="pretrained_eval_metrics.json",
        save=True,
    )

    # --- Optional: visualise the first frame of each val sequence ---
    if SAVE_VIS:
        os.makedirs(VIS_DIR, exist_ok=True)
        print(f"\nSaving visualisations to {VIS_DIR} ...")

        for seq_idx in VAL_SEQS:
            img_path = Path(DATASET_PATH) / f"{seq_idx:04d}" / "000000.png"
            if not img_path.exists():
                continue

            image         = Image.open(img_path).convert("RGB")
            unique_img_id = seq_idx * 100000   # frame 0

            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
            preds = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )[0]

            # Predictions (red) — filter to person/car only
            valid_boxes, valid_labels, valid_scores = filter_results(
                preds["scores"], preds["labels"], preds["boxes"]
            )

            # Ground truth (green) and ignore regions (orange)
            ann_ids  = coco_gt.getAnnIds(imgIds=[unique_img_id])
            anns     = coco_gt.loadAnns(ann_ids)
            gt_boxes,  gt_labels  = [], []
            ign_boxes, ign_labels = [], []
            for ann in anns:
                if ann.get("iscrowd") == 1:
                    ign_boxes.append(ann["bbox"])
                    ign_labels.append(ann["category_id"])
                else:
                    gt_boxes.append(ann["bbox"])
                    gt_labels.append(ann["category_id"])

            if valid_boxes:
                image = draw_bboxes(image, valid_boxes, valid_labels, valid_scores,
                                    label_map=COCO_LABELS, box_type="pred")
            if gt_boxes:
                image = draw_bboxes(image, gt_boxes, gt_labels,
                                    label_map={1: "person", 3: "car"}, box_type="gt")
            if ign_boxes:
                image = draw_bboxes(image, ign_boxes, ign_labels,
                                    label_map={1: "person", 3: "car"}, box_type="ignore")

            image.save(os.path.join(VIS_DIR, f"seq_{seq_idx:04d}_pretrained.png"))

        print("Visualisations saved.")


if __name__ == "__main__":
    run_evaluation()
