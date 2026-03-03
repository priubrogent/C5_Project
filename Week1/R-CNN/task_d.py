import os
import json
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F
from tqdm import tqdm
from pycocotools.coco import COCO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, draw_bboxes, filter_results, COCO_CLASSES, TRAIN_SEQS, VAL_SEQS

DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_d/"
VIS_DIR = "./R-CNN/Results_RCNN/task_d/visualizations"
GT_PATH = "kitti_mots_to_coco_gt.json"
N_FOLDERS = 21

def visualize_first_frames():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(VIS_DIR, exist_ok=True)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    categories = weights.meta["categories"]
    COCO_LABELS = {i: categories[i] for i in range(len(categories))}

    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found.")
        return
    coco_gt = COCO(GT_PATH)

    print(f"Generating visualizations for the first frame of validation sequences...")

    for seq_idx in VAL_SEQS:
        img_path = Path(DATASET_PATH) / f"{seq_idx:04d}" / "000000.png"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        unique_image_id = (seq_idx * 100000)

        img_tensor = F.to_tensor(image).to(device)
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        preds = {
            "scores": outputs["scores"],
            "labels": outputs["labels"],
            "boxes": outputs["boxes"]
        }

        ann_ids = coco_gt.getAnnIds(imgIds=[unique_image_id])
        anns = coco_gt.loadAnns(ann_ids)

        gt_boxes, gt_labels = [], []
        ign_boxes, ign_labels = [], []

        for ann in anns:
            if ann.get('iscrowd') == 1 or ann.get('category_id') == 10:
                ign_boxes.append(ann['bbox'])
                ign_labels.append(ann['category_id'])
            else:
                gt_boxes.append(ann['bbox'])
                gt_labels.append(ann['category_id'])


        valid_boxes, valid_labels, valid_scores = filter_results(preds["scores"], preds["labels"], preds["boxes"])

        if len(valid_boxes) > 0:
            image = draw_bboxes(image, valid_boxes, valid_labels, valid_scores, label_map=COCO_LABELS, threshold=0.5, box_type="pred")

        if gt_boxes:
            image = draw_bboxes(image, gt_boxes, gt_labels, label_map=COCO_LABELS, box_type="gt")

        if ign_boxes:
            image = draw_bboxes(image, ign_boxes, ign_labels, label_map=COCO_LABELS, box_type="ignore")

        save_path = os.path.join(VIS_DIR, f"seq_{seq_idx:04d}_vis.png")
        image.save(save_path)
        print(f"Saved: {save_path}")

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    results_list = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found. Run GT_conversor.py first.")
        return
    coco_gt = COCO(GT_PATH)
    valid_ids = set(coco_gt.getImgIds())

    print(f"Starting evaluation on {N_FOLDERS} sequences...")

    for seq_idx in tqdm(VAL_SEQS, desc="Sequences", position=0):
        folder = Path(DATASET_PATH) / f"{seq_idx:04d}"
        if not folder.exists(): continue

        img_files = sorted(list(folder.glob("*.png")))

        for img_path in tqdm(img_files, desc=f"Seq {seq_idx:04d}", position=1, leave=False):
            frame_idx = int(img_path.stem)
            unique_image_id = (seq_idx * 100000) + frame_idx

            if unique_image_id not in valid_ids:
                continue

            image = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(image).to(device)

            with torch.no_grad():
                outputs = model([img_tensor])[0]

            results = {
                "scores": outputs["scores"],
                "labels": outputs["labels"],
                "boxes": outputs["boxes"]
            }

            for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
                label_id = label.item()
                if label_id in COCO_CLASSES:
                    x1, y1, x2, y2 = bbox.tolist()
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                    results_list.append({
                        "image_id": unique_image_id,
                        "category_id": label_id,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    coco_evaluation(results_list, coco_gt, OUTPUT_DIR)

if __name__ == "__main__":
    run_evaluation()
    visualize_first_frames()
