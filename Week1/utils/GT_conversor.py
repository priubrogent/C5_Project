import os
import json
from pathlib import Path
from pycocotools import mask as maskUtils
from utils import KITTI_TO_COCO

# --- Configuration ---
ANNOTATIONS_DIR = "/ghome/mcv/datasets/C5/KITTI-MOTS/instances_txt"
OUTPUT_FILE = "kitti_mots_to_coco_gt.json"
SEQ_RANGE = 21 

def convert_kitti_mots_to_coco():
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"}
        ]
    }

    ann_id = 0
    # Optimization: Use a set to track image IDs instantly
    seen_images = set()

    for i in range(SEQ_RANGE):
        txt_path = Path(ANNOTATIONS_DIR) / f"{i:04d}.txt"
        if not txt_path.exists():
            continue

        print(f"Processing sequence {i:04d}...")
        
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                # Format: [frame, id, class, height, width, rle]
                frame_idx = int(parts[0])
                obj_id = int(parts[1])
                h, w = int(parts[3]), int(parts[4])
                rle_str = parts[5]

                # 1. Unique Image ID logic
                unique_img_id = (i * 100000) + frame_idx
                
                # Add image info only if new (O(1) lookup)
                if unique_img_id not in seen_images:
                    coco_data['images'].append({
                        "id": unique_img_id,
                        "file_name": f"{i:04d}/{frame_idx:06d}.png",
                        "height": h,
                        "width": w
                    })
                    seen_images.add(unique_img_id)

                # 2. Extract Class and Instance IDs
                extracted_class_id = obj_id // 1000
                instance_id = obj_id % 1000 
                
                # 3. Handle Mapping and Ignore Regions
                coco_class_id = KITTI_TO_COCO.get(extracted_class_id)
                
                # ID 10 (10000) is ignore region
                is_crowd = 1 if extracted_class_id == 10 else 0
                
                if coco_class_id is None and not is_crowd:
                    continue

                # 4. Process Mask and BBox
                rle_dict = {"size": [h, w], "counts": rle_str.encode('utf-8')}
                bbox = maskUtils.toBbox(rle_dict).tolist() # [x, y, w, h]
                area = maskUtils.area(rle_dict).item()

                coco_data['annotations'].append({
                    "id": ann_id,
                    "image_id": unique_img_id,
                    # Map ignore regions to 'person' (1) but with iscrowd=1 so they don't hurt metrics
                    "category_id": coco_class_id if coco_class_id else 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": is_crowd,
                    "segmentation": rle_dict["counts"].decode('utf-8'),
                })
                ann_id += 1

    with open(OUTPUT_FILE, "w") as f:
        json.dump(coco_data, f)
    print(f"Finished! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_kitti_mots_to_coco()