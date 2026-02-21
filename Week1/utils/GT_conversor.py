import os
import json
from pathlib import Path
from pycocotools import mask as maskUtils
# Assuming KITTI_TO_COCO is {1: 1, 2: 3} (Pedestrian -> Person, Car -> Car)
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
    seen_images = set()

    for i in range(SEQ_RANGE):
        txt_path = Path(ANNOTATIONS_DIR) / f"{i:04d}.txt"
        if not txt_path.exists():
            continue

        print(f"Processing sequence {i:04d}...")
        
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                frame_idx = int(parts[0])
                obj_id = int(parts[1])
                h, w = int(parts[3]), int(parts[4])
                rle_str = parts[5]

                unique_img_id = (i * 100000) + frame_idx
                
                if unique_img_id not in seen_images:
                    coco_data['images'].append({
                        "id": unique_img_id,
                        "file_name": f"{i:04d}/{frame_idx:06d}.png",
                        "height": h,
                        "width": w
                    })
                    seen_images.add(unique_img_id)

                extracted_class_id = obj_id // 1000
                coco_class_id = KITTI_TO_COCO.get(extracted_class_id)
                
                # KITTI Class 10 is the "Ignore" region
                is_crowd = 1 if extracted_class_id == 10 else 0
                
                if coco_class_id is None and not is_crowd:
                    continue

                # Process Mask and BBox
                rle_dict = {"size": [h, w], "counts": rle_str.encode('utf-8')}
                bbox = maskUtils.toBbox(rle_dict).tolist() 
                area = float(maskUtils.area(rle_dict))

                # Logic: If it's an ignore region, we add it for BOTH classes.
                # This prevents False Positive penalties for either class.
                target_classes = [1, 3] if is_crowd else [coco_class_id]

                for cls_id in target_classes:
                    coco_data['annotations'].append({
                        "id": ann_id,
                        "image_id": unique_img_id,
                        "category_id": cls_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": is_crowd,
                        # Store as string for JSON serializability
                        "segmentation": rle_str 
                    })
                    ann_id += 1

    with open(OUTPUT_FILE, "w") as f:
        json.dump(coco_data, f)
    print(f"Finished! Total annotations: {ann_id}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_kitti_mots_to_coco()