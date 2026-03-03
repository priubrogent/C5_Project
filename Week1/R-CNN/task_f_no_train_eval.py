import os
import sys
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as F
from tqdm import tqdm
from datasets import load_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OUTPUT_DIR = "./R-CNN/Results_RCNN/task_f_no_train/"
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_SPLIT = 0.2
SEED = 42

DEART_PERSON_CLASSES = {"person", "nude"}
COCO_PERSON_ID = 1


class DEArtDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self._build_category_id_to_name()

    def _build_category_id_to_name(self):
        self.cat_id_to_name = {}

        sample_size = min(100, len(self.dataset))
        print(f"Sampling {sample_size} items to build category mapping...")
        for i in range(sample_size):
            item = self.dataset[i]
            ann_data = json.loads(item['annotations'])
            for cat in ann_data.get('categories', []):
                if cat['id'] not in self.cat_id_to_name:
                    self.cat_id_to_name[cat['id']] = cat['name'].lower()

        self.person_cat_ids = set()
        for cat_id, name in self.cat_id_to_name.items():
            if name in DEART_PERSON_CLASSES:
                self.person_cat_ids.add(cat_id)

        print(f"Found person-related category IDs: {self.person_cat_ids}")
        print(f"Category mapping: {[(cid, self.cat_id_to_name[cid]) for cid in self.person_cat_ids]}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

        ann_data = json.loads(item['annotations'])
        annotations = ann_data.get('annotations', [])

        local_cat_id_to_name = {}
        for cat in ann_data.get('categories', []):
            local_cat_id_to_name[cat['id']] = cat['name'].lower()

        bboxes = []
        class_labels = []

        for ann in annotations:
            if ann.get('iscrowd', 0) == 0:
                cat_id = ann['category_id']
                cat_name = local_cat_id_to_name.get(cat_id, '').lower()

                if cat_name in DEART_PERSON_CLASSES:
                    bbox = ann['bbox']
                    bboxes.append(bbox)
                    class_labels.append(COCO_PERSON_ID)

        image_tensor = F.to_tensor(image_np)

        boxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(class_labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "bboxes_coco": bboxes,
        }

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_coco_gt_and_run_inference(dataset, model, device):
    images = []
    annotations = []
    results_list = []
    ann_id = 1

    model.eval()

    print("Building GT and running inference in single pass...")

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing"):
            image_tensor, target = dataset[idx]
            img_id = idx

            images.append({
                "id": img_id,
                "file_name": f"{img_id}.jpg",
                "height": 1,
                "width": 1,
            })

            for bbox, label in zip(target["bboxes_coco"], target["labels"].tolist()):
                x, y, w, h = bbox
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": label,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

            image_tensor = image_tensor.to(device)
            outputs = model([image_tensor])[0]

            for score, label, bbox in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
                label_id = label.item()
                if label_id == COCO_PERSON_ID:
                    x1, y1, x2, y2 = bbox.tolist()
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]
                    results_list.append({
                        "image_id": img_id,
                        "category_id": COCO_PERSON_ID,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}]
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    gt_path = os.path.join(OUTPUT_DIR, "deart_gt_coco.json")
    with open(gt_path, "w") as f:
        json.dump(coco_dict, f)

    coco_gt = COCO(gt_path)

    return coco_gt, results_list


def coco_evaluation_single_class(results_list, coco_gt, output_path):
    print("\n--- Running COCO Evaluation (Person class only) ---")

    if len(results_list) == 0:
        print("Warning: No detections to evaluate!")
        return {}

    coco_dt = coco_gt.loadRes(results_list)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    coco_eval.params.catIds = [COCO_PERSON_ID]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats.tolist()
    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75", "mAP_s", "mAP_m", "mAP_l",
        "mAR_1", "mAR_10", "mAR_100", "mAR_s", "mAR_m", "mAR_l"
    ]
    final_metrics = {name: round(float(stat), 4) for name, stat in zip(metric_names, stats)}

    output_file = os.path.join(output_path, "evaluation_metrics.json")
    with open(output_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {output_file}")

    return final_metrics


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading DEArt dataset from HuggingFace...")
    full_dataset = load_dataset("biglam/european_art", split="train")
    print(f"Total dataset size: {len(full_dataset)}")

    print(f"Splitting dataset: {int((1-VAL_SPLIT)*100)}% train / {int(VAL_SPLIT*100)}% val")
    split_dataset = full_dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    val_hf_dataset = split_dataset["test"]
    print(f"Validation split size: {len(val_hf_dataset)}")

    print("Creating evaluation dataset (person/nude only)...")
    eval_dataset = DEArtDataset(val_hf_dataset)

    print("Loading pre-trained Faster R-CNN (COCO weights)...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    categories = weights.meta["categories"]
    print(f"COCO person class index: {categories.index('person')}")

    coco_gt, results_list = build_coco_gt_and_run_inference(eval_dataset, model, device)

    print(f"Ground truth annotations: {len(coco_gt.getAnnIds())}")
    print(f"Total person detections: {len(results_list)}")

    pred_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results_list, f)
    print(f"Predictions saved to: {pred_path}")

    metrics = coco_evaluation_single_class(results_list, coco_gt, OUTPUT_DIR)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY (Pre-trained model on DEArt)")
    print("="*50)
    print(f"Dataset: DEArt (European Art) - Validation Split ({int(VAL_SPLIT*100)}%)")
    print(f"Classes evaluated: person, nude -> COCO person")
    print(f"Total images: {len(eval_dataset)}")
    print(f"GT annotations: {len(coco_gt.getAnnIds())}")
    print(f"Total detections: {len(results_list)}")
    print("-"*50)
    print(f"mAP@0.50:0.95 = {metrics.get('mAP_0.50_0.95', 'N/A')}")
    print(f"mAP@0.50      = {metrics.get('mAP_0.50', 'N/A')}")
    print(f"mAP@0.75      = {metrics.get('mAP_0.75', 'N/A')}")
    print(f"mAR@100       = {metrics.get('mAR_100', 'N/A')}")
    print("="*50)

    return metrics


if __name__ == "__main__":
    run_evaluation()
