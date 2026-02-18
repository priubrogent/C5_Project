import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torchvision.transforms.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pycocotools import mask as mask_util
import os
from kitti_coco_mapping import KITTI_TO_COCO, COCO_CLASSES

DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS"
IMAGES_PATH = os.path.join(DATASET_PATH, "testing/image_02")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "instances_txt")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def load_annotations(ann_path, frame_id):
    boxes, labels = [], []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != frame_id:
                continue
            obj_id = int(parts[1])
            class_id = obj_id // 1000
            if class_id not in KITTI_TO_COCO:
                continue
            h, w = int(parts[3]), int(parts[4])
            rle = {"counts": parts[5].encode(), "size": [h, w]}
            x, y, bw, bh = mask_util.toBbox(rle)
            boxes.append([x, y, x + bw, y + bh])
            labels.append(KITTI_TO_COCO[class_id])
    return boxes, labels

sequences = sorted(os.listdir(IMAGES_PATH))
all_metric = MeanAveragePrecision()

for seq in sequences:
    seq_images_path = os.path.join(IMAGES_PATH, seq)
    ann_file = os.path.join(ANNOTATIONS_PATH, f"{seq}.txt")
    if not os.path.exists(ann_file):
        continue

    seq_metric = MeanAveragePrecision()
    images = sorted(os.listdir(seq_images_path))
    print(f"seq {seq} ({len(images)} frames)")

    for i, img_name in enumerate(images):
        image = Image.open(os.path.join(seq_images_path, img_name)).convert("RGB")
        img_tensor = F.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])[0]

        mask = torch.tensor([l.item() in COCO_CLASSES for l in prediction["labels"]])
        preds = [{
            "boxes": prediction["boxes"][mask].cpu(),
            "scores": prediction["scores"][mask].cpu(),
            "labels": prediction["labels"][mask].cpu(),
        }]

        boxes, labels = load_annotations(ann_file, i)
        targets = [{
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }]

        seq_metric.update(preds, targets)
        all_metric.update(preds, targets)

    r = seq_metric.compute()
    print(f"  mAP: {r['map']:.4f}  mAP@50: {r['map_50']:.4f}  mAP@75: {r['map_75']:.4f}")

r = all_metric.compute()

print(f"TOTS:  mAP: {r['map']:.4f}  mAP@50: {r['map_50']:.4f}  mAP@75: {r['map_75']:.4f}")