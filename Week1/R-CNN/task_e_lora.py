import os
import random
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from tqdm import tqdm
from pycocotools.coco import COCO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, COCO_CLASSES, VAL_SEQS, TRAIN_SEQS, COCO_TO_RCNN_ID, RCNN_TO_COCO_ID

SEED = 42
DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_e_lora/"

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        for param in self.original_layer.parameters():
            param.requires_grad = False

        if isinstance(original_layer, nn.Conv2d):
            self.in_features = original_layer.in_channels
            self.out_features = original_layer.out_channels
            kernel_size = original_layer.kernel_size
            stride = original_layer.stride
            padding = original_layer.padding

            self.lora_A = nn.Conv2d(
                self.in_features, rank,
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.lora_B = nn.Conv2d(
                rank, self.out_features,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        elif isinstance(original_layer, nn.Linear):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features

            self.lora_A = nn.Linear(self.in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        else:
            raise ValueError(f"LoRA not supported for {type(original_layer)}")

        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return original_output + lora_output * self.scaling


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.1, target_modules=None):
    if target_modules is None:
        target_modules = ["backbone.body.layer3", "backbone.body.layer4"]

    lora_layers = []

    def replace_with_lora(parent, name, module):
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            lora_layer = LoRALayer(module, rank, alpha, dropout)
            setattr(parent, name, lora_layer)
            lora_layers.append(lora_layer)
            return True
        return False

    def recursive_apply(parent, prefix=""):
        for name, module in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            should_apply = any(target in full_name for target in target_modules)

            if should_apply and isinstance(module, nn.Conv2d):
                if replace_with_lora(parent, name, module):
                    print(f"  Applied LoRA to: {full_name}")
            else:
                recursive_apply(module, full_name)

    recursive_apply(model)

    return lora_layers


def freeze_non_lora_params(model):
    lora_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        if "lora_" in name or "box_predictor" in name:
            param.requires_grad = True
            lora_params += param.numel()
        else:
            param.requires_grad = False
            other_params += param.numel()

    total = lora_params + other_params
    print(f"\nParameter count:")
    print(f"  Trainable (LoRA + head): {lora_params:,} ({100*lora_params/total:.2f}%)")
    print(f"  Frozen: {other_params:,} ({100*other_params/total:.2f}%)")

    return lora_params


class KittiMotsDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, sequence_ids, transform=None):
        super().__init__(img_folder, ann_file)
        self.transform = transform

        self.ids = [
            idx for idx in self.ids
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]

        val_img_ids_set = set(self.ids)

        self.coco.dataset['images'] = [
            img for img in self.coco.dataset['images'] if img['id'] in val_img_ids_set
        ]
        self.coco.dataset['annotations'] = [
            ann for ann in self.coco.dataset['annotations'] if ann['image_id'] in val_img_ids_set
        ]

        self.coco.createIndex()

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]

        image = cv2.imread(os.path.join(self.root, img_metadata['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        bboxes = []
        class_labels = []
        for ann in target:
            cat_id = ann['category_id']
            if cat_id in COCO_CLASSES and ann.get('iscrowd', 0) == 0:
                bboxes.append(ann['bbox'])
                class_labels.append(COCO_TO_RCNN_ID[cat_id])

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        image = F.to_tensor(image)

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
            "image_id": torch.tensor([img_id])
        }

        return image, target


def collate_fn(batch):
    batch = [b for b in batch if len(b[1]["boxes"]) > 0]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


def evaluate_model(model, coco_gt, device):
    model.eval()
    valid_ids = set(coco_gt.getImgIds())
    results_list = []

    for seq_idx in tqdm(VAL_SEQS, desc="Evaluating"):
        folder = Path(DATASET_PATH) / f"{seq_idx:04d}"
        if not folder.exists():
            continue

        img_files = sorted(list(folder.glob("*.png")))

        for img_path in img_files:
            frame_idx = int(img_path.stem)
            unique_image_id = (seq_idx * 100000) + frame_idx

            if unique_image_id not in valid_ids:
                continue

            image = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(image).to(device)

            with torch.no_grad():
                outputs = model([img_tensor])[0]

            for score, label, bbox in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
                label_id = label.item()
                if label_id in RCNN_TO_COCO_ID:
                    coco_label = RCNN_TO_COCO_ID[label_id]
                    x1, y1, x2, y2 = bbox.tolist()
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                    results_list.append({
                        "image_id": unique_image_id,
                        "category_id": coco_label,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    return results_list


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading Faster R-CNN...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print("\nApplying LoRA to backbone layers...")
    lora_layers = apply_lora_to_model(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        target_modules=["backbone.body.layer3", "backbone.body.layer4"]
    )
    print(f"Created {len(lora_layers)} LoRA layers")

    trainable_params = freeze_non_lora_params(model)

    model.to(device)

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))

    train_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, TRAIN_SEQS, transform=train_transforms)
    val_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, VAL_SEQS, transform=None)

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\nStarting LoRA training...")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Train"):
            if batch is None:
                continue

            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.train()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Val"):
                if batch is None:
                    continue

                images, targets = batch
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))

    import json
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "trainable_params": trainable_params,
    }
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print("Running final evaluation...")
    print("="*60)

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()

    coco_gt = COCO(ANNOTATION_FILE)

    from PIL import Image

    results_list = evaluate_model(model, coco_gt, device)
    coco_evaluation(results_list, coco_gt, OUTPUT_DIR)

    print(f"\nLoRA training complete!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
