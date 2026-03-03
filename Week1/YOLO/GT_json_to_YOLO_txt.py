import json
import os
import cv2
from pathlib import Path
from pycocotools.coco import COCO

from Week1.utils.utils import TRAIN_SEQS, VAL_SEQS
import shutil
from tqdm import tqdm

DATASET_ROOT = "/data/113-2/users/gasbert/master/C5/KITTI-MOTS"
INPUT_IMAGES_PATH = os.path.join(DATASET_ROOT, "training/image_02")
INPUT_ANN_PATH = os.path.join(DATASET_ROOT, "kitti_mots_to_coco_gt.json")

OUTPUT_FOLDER = os.path.join(DATASET_ROOT, "yolo_data")
TRAINING_FOLDER = os.path.join(OUTPUT_FOLDER, "train")
VALIDATION_FOLDER = os.path.join(OUTPUT_FOLDER, "val")
OUTPUT_TRAIN_IMAGES = os.path.join(TRAINING_FOLDER, "images")
OUTPUT_VAL_IMAGES = os.path.join(VALIDATION_FOLDER, "images")
OUTPUT_TRAIN_LABELS = os.path.join(TRAINING_FOLDER, "labels")
OUTPUT_VAL_LABELS = os.path.join(VALIDATION_FOLDER, "labels")

coco = COCO(INPUT_ANN_PATH)

os.makedirs(OUTPUT_TRAIN_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_VAL_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_TRAIN_LABELS, exist_ok=True)
os.makedirs(OUTPUT_VAL_LABELS, exist_ok=True)

for img_id in tqdm(coco.getImgIds()):

    img_info = coco.loadImgs(img_id)[0]

    # Recover sequence and frame from your ID logic
    seq_idx = img_id // 100000
    frame_idx = img_id % 100000

    og_img_path = Path(INPUT_IMAGES_PATH) / f"{seq_idx:04d}" / f"{frame_idx:06d}.png"

    if seq_idx in TRAIN_SEQS:
        img_path = Path(OUTPUT_TRAIN_IMAGES) / f"{seq_idx:04d}" / f"{frame_idx:06d}.png"
    elif seq_idx in VAL_SEQS:
        img_path = Path(OUTPUT_VAL_IMAGES) / f"{seq_idx:04d}" / f"{frame_idx:06d}.png"
    else:
        continue

    img_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(og_img_path), str(img_path))

    image = cv2.imread(str(img_path))
    h, w = image.shape[:2]

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    if seq_idx in TRAIN_SEQS:
        label_dir = Path(OUTPUT_TRAIN_LABELS) / f"{seq_idx:04d}"
    elif seq_idx in VAL_SEQS:
        label_dir = Path(OUTPUT_VAL_LABELS) / f"{seq_idx:04d}"
    else:
        continue

    label_dir.mkdir(parents=True, exist_ok=True)

    label_path = label_dir / f"{frame_idx:06d}.txt"

    with open(label_path, "w") as f:
        for ann in anns:

            # Skip ignore regions
            if ann.get("iscrowd") == 1:
                continue

            coco_cls = ann["category_id"]

            # Map COCO → YOLO class indices
            if coco_cls == 1:      # Pedestrian
                yolo_cls = 0
            elif coco_cls == 3:    # Car
                yolo_cls = 1
            else:
                continue

            x, y, bw, bh = ann["bbox"]

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            f.write(f"{yolo_cls} {x_center} {y_center} {bw} {bh}\n")