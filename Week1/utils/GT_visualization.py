import os
import json
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

# Import the refactored universal drawing function
from utils import draw_bboxes

# --- Hyperparameters ---
DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
GT_PATH = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./DeTR/Results_DETR/ground_truth_viz/"

# Visualization Settings
N_SEQUENCES = 21             # Number of sequences to process (Max 21)
SHOW_IGNORE_REGIONS = True  # Set to True to draw ignore region boxes in orange

# Label mapping for visualization
LABEL_MAP = {1: "person", 3: "car", 10: "ignore"}

def generate_gt_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found.")
        return
    
    # Load COCO Ground Truth
    coco_gt = COCO(GT_PATH)
    all_img_ids = sorted(coco_gt.getImgIds())

    print(f"Generating GT visualization for the first image of {N_SEQUENCES} sequences...")

    # Iterate through the sequences
    for seq_idx in range(N_SEQUENCES):
        # Filter all image IDs belonging to the current sequence
        # ID Logic: (seq_idx * 100000) + frame_idx
        seq_start = seq_idx * 100000
        seq_end = seq_start + 99999
        seq_img_ids = [i for i in all_img_ids if seq_start <= i <= seq_end]
        
        if not seq_img_ids:
            print(f"Skipping Seq {seq_idx:04d}: No annotations found.")
            continue
            
        # Select the first image (frame) available in the sequence
        img_id = seq_img_ids[0]
        
        # Load Image metadata
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        
        img_path = Path(DATASET_PATH) / file_name
        if not img_path.exists():
            print(f"Skipping {file_name}: File not found.")
            continue
            
        # 1. Use PIL as required by the draw_bboxes function
        image = Image.open(img_path).convert("RGB")
        
        # 2. Separate valid annotations and ignore regions
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        
        valid_boxes, valid_labels = [], []
        ignore_boxes, ignore_labels = [], []
        
        for ann in anns:
            # ID 10,000 denotes an ignore region
            is_ignore = ann.get('iscrowd') == 1 or ann.get('category_id') == 10
            
            if is_ignore:
                ignore_boxes.append(ann['bbox'])
                ignore_labels.append(ann['category_id'])
            else:
                valid_boxes.append(ann['bbox'])
                valid_labels.append(ann['category_id'])

        # 3. Draw using the draw_bboxes function
        # Draw Valid GT (Green)
        if valid_boxes:
            image = draw_bboxes(
                image, valid_boxes, valid_labels, 
                label_map=LABEL_MAP, box_type="gt"
            )
            
        # Draw Ignore Regions (Orange)
        if SHOW_IGNORE_REGIONS and ignore_boxes:
            image = draw_bboxes(
                image, ignore_boxes, ignore_labels, 
                label_map=LABEL_MAP, box_type="ignore"
            )

        # 4. Save result with a clear sequence-based filename
        save_name = f"seq_{seq_idx:04d}_first_frame.png"
        image.save(os.path.join(OUTPUT_DIR, save_name))
        print(f"Saved: {save_name}")

    print(f"\nFinished! Visualized images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_gt_images()