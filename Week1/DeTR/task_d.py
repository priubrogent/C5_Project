import os
import json
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from pycocotools.coco import COCO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, draw_bboxes, filter_results, COCO_CLASSES

# --- Configuration ---
DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_DIR = "./DeTR/Results_DETR/task_d/"
VIS_DIR = "./DeTR/Results_DETR/task_d/visualizations"
GT_PATH = "kitti_mots_to_coco_gt.json"
N_FOLDERS = 21 # Number of folders to process (21 maximum)

def visualize_first_frames():
    """
    Inferences the first image of each sequence and overlays Pred, GT, and Ignore boxes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(VIS_DIR, exist_ok=True)

    # Initialize model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()
    
    COCO_LABELS = model.config.id2label

    # Load Ground Truth
    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found.")
        return
    coco_gt = COCO(GT_PATH)

    print(f"Generating visualizations for the first frame of {N_FOLDERS} sequences...")

    for seq_idx in range(N_FOLDERS):
        # 1. Pathing and Image Loading
        # In KITTI-MOTS, the first frame is always 000000.png
        img_path = Path(DATASET_PATH) / f"{seq_idx:04d}" / "000000.png"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        unique_image_id = (seq_idx * 100000) # frame_idx is 0

        # 2. Run Inference for Predictions
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
        # Use threshold=0.5 for clean visualization, threshold=0 is only for mAP
        preds = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

        # 3. Fetch Ground Truth and Ignore Regions
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
                
                
        # Filter results to only include valid classes and prepare for drawing
        valid_boxes, valid_labels, valid_scores = filter_results(preds["scores"], preds["labels"], preds["boxes"]) 

        # 4. Draw using draw_bboxes
        # Draw Predictions (Red)
        if len(valid_boxes) > 0:
            image = draw_bboxes(image, valid_boxes, valid_labels, valid_scores, label_map=COCO_LABELS, box_type="pred")

        # Draw Valid GT (Green)
        if gt_boxes:
            image = draw_bboxes(image, gt_boxes, gt_labels, label_map=COCO_LABELS, box_type="gt")

        # Draw Ignore Regions (Orange)
        if ign_boxes:
            image = draw_bboxes(image, ign_boxes, ign_labels, label_map=COCO_LABELS, box_type="ignore")

        # 5. Save output
        save_path = os.path.join(VIS_DIR, f"seq_{seq_idx:04d}_vis.png")
        image.save(save_path)
        print(f"Saved: {save_path}")

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    model.eval()

    results_list = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Pre-load GT to verify we only process valid images
    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found. Run GT_conversor.py first.")
        return
    coco_gt = COCO(GT_PATH)
    valid_ids = set(coco_gt.getImgIds())

    print(f"Starting evaluation on {N_FOLDERS} sequences...")

    for seq_idx in tqdm(range(N_FOLDERS), desc="Sequences", position=0):
        # Construct folder path and check existence
        folder = Path(DATASET_PATH) / f"{seq_idx:04d}"
        if not folder.exists(): continue
        
        img_files = sorted(list(folder.glob("*.png")))

        for img_path in tqdm(img_files, desc=f"Seq {seq_idx:04d}", position=1, leave=False):
            frame_idx = int(img_path.stem)
            # Generate unique image_id matching GT logic
            unique_image_id = (seq_idx * 100000) + frame_idx
            
            # Skip images not in GT (saves time and avoids errors)
            if unique_image_id not in valid_ids:
                continue

            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Explicitly construct target_sizes as (Height, Width) for correct scaling
            target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
            
            # Use threshold=0 to get all queries for proper mAP calculation
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0)[0]

            # Filter and store results for COCO evaluation
            for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
                label_id = label.item()
                # Consider only valid classes (person and car) for evaluation
                if label_id in COCO_CLASSES:
                    x1, y1, x2, y2 = bbox.tolist()
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1] # Convert to COCO format [x, y, w, h]

                    results_list.append({
                        "image_id": unique_image_id,
                        "category_id": label_id,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    # --- Run COCO Evaluation ---
    coco_evaluation(results_list, coco_gt, OUTPUT_DIR)

if __name__ == "__main__":
    #run_evaluation()
    visualize_first_frames()