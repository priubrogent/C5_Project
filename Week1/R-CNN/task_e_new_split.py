"""
task_e_new_split.py — Faster R-CNN fine-tuning with an alternative sequence split.

Purpose: Test whether the ~0.1 gap between val loss < train loss is caused by the
specific sequence assignment in the original KITTI-MOTS paper split, or by other
factors (e.g. data augmentation making training harder, batch-size differences).

How the split is generated:
  All 21 sequences (0-20) are randomly shuffled with seed=99 (different from the
  training seed of 42). The first 12 become the new training set, the last 9 become
  the new validation set — preserving the original 12/9 ratio and keeping sequence
  boundaries intact (no temporal leakage).

Expected outcome:
  - If the gap shrinks considerably  -> original sequence choice was biased (easier
    sequences landed in VAL_SEQS).
  - If the gap stays roughly the same -> data augmentation is the dominant cause
    (training loss is structurally higher because augmented images are harder).
"""

import os
import random
import sys
import numpy as np
import torch
import albumentations as A
import cv2
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, COCO_CLASSES, COCO_TO_RCNN_ID, RCNN_TO_COCO_ID
from utils.notify import notify

# ── New split (seed=99) ────────────────────────────────────────────────────────
_all_seqs = list(range(21))
_rng = random.Random(99)
_rng.shuffle(_all_seqs)
NEW_TRAIN_SEQS = sorted(_all_seqs[:12])
NEW_VAL_SEQS   = sorted(_all_seqs[12:])
print(f"[split] NEW_TRAIN_SEQS = {NEW_TRAIN_SEQS}")
print(f"[split] NEW_VAL_SEQS   = {NEW_VAL_SEQS}")

# Original paper split (kept for reference / easy comparison)
ORIG_TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
ORIG_VAL_SEQS   = [2, 6, 7, 8, 10, 13, 14, 16, 18]

# --- Configuration ---
SEED         = 42
DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR   = "./R-CNN/Results_RCNN/task_e_new_split/"

# --- Hyperparameters (same as task_e) ---
NUM_EPOCHS    = 20
BATCH_SIZE    = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 5e-4
NUM_WORKERS   = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KittiMotsDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, sequence_ids, transform=None):
        super().__init__(img_folder, ann_file)
        self.transform = transform

        self.ids = [
            idx for idx in self.ids
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]

        img_id_set = set(self.ids)
        self.coco.dataset['images'] = [
            img for img in self.coco.dataset['images'] if img['id'] in img_id_set
        ]
        self.coco.dataset['annotations'] = [
            ann for ann in self.coco.dataset['annotations'] if ann['image_id'] in img_id_set
        ]
        self.coco.createIndex()

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        image = cv2.imread(os.path.join(self.root, img_metadata['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target  = self.coco.loadAnns(ann_ids)

        bboxes, class_labels = [], []
        for ann in target:
            cat_id = ann['category_id']
            if cat_id in COCO_CLASSES and ann.get('iscrowd', 0) == 0:
                bboxes.append(ann['bbox'])
                class_labels.append(COCO_TO_RCNN_ID[cat_id])

        if self.transform:
            transformed    = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image          = transformed['image']
            bboxes         = transformed['bboxes']
            class_labels   = transformed['class_labels']

        image = F.to_tensor(image)

        boxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.as_tensor(boxes,        dtype=torch.float32)
            labels = torch.as_tensor(class_labels, dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}


def collate_fn(batch):
    batch = [b for b in batch if len(b[1]["boxes"]) > 0]
    return tuple(zip(*batch)) if batch else None


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="kitti-mots-rcnn",
        name="faster-rcnn-new-split",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "split_seed": 99,
            "train_seqs": NEW_TRAIN_SEQS,
            "val_seqs": NEW_VAL_SEQS,
            "note": "Testing whether sequence split causes val_loss < train_loss gap",
        }
    )

    # Model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)
    model.to(device)

    # Augmentations (identical to task_e — only the split changes)
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'], min_visibility=0.3))

    train_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, NEW_TRAIN_SEQS, transform=train_transforms)
    val_dataset   = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, NEW_VAL_SEQS,   transform=None)

    print(f"Train dataset size : {len(train_dataset)}  (seqs {NEW_TRAIN_SEQS})")
    print(f"Val   dataset size : {len(val_dataset)}   (seqs {NEW_VAL_SEQS})")
    notify(
        f"Training started (new split, seed=99)\ntrain={len(train_dataset)} imgs  val={len(val_dataset)} imgs\n"
        f"train_seqs={NEW_TRAIN_SEQS}\nval_seqs={NEW_VAL_SEQS}",
        title="R-CNN Task E (new split)",
        tags=["rocket"],
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Train"):
            if batch is None:
                continue
            images, targets = batch
            images  = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_loss = losses.item()
            epoch_loss += batch_loss
            global_step += 1

            wandb.log({
                "batch/loss_total":       batch_loss,
                "batch/loss_classifier":  loss_dict.get("loss_classifier",  torch.tensor(0.0)).item(),
                "batch/loss_box_reg":     loss_dict.get("loss_box_reg",     torch.tensor(0.0)).item(),
                "batch/loss_objectness":  loss_dict.get("loss_objectness",  torch.tensor(0.0)).item(),
                "batch/loss_rpn_box_reg": loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item(),
            }, step=global_step)

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.train()  # keep train mode so Faster R-CNN returns losses
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Val"):
                if batch is None:
                    continue
                images, targets = batch
                images  = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        gap = avg_train_loss - avg_val_loss
        print(f"Epoch {epoch+1:02d} | train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  gap={gap:+.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "gap_train_minus_val": gap,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            wandb.run.summary["best_val_loss"] = best_val_loss
            notify(
                f"Epoch {epoch+1}/{NUM_EPOCHS} — new best!\ntrain={avg_train_loss:.4f}  val={avg_val_loss:.4f}  gap={gap:+.4f}",
                title="R-CNN Task E (new split)",
                tags=["star"],
            )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    notify(
        f"Training complete!\nbest_val_loss={best_val_loss:.4f}\nResults: {OUTPUT_DIR}",
        title="R-CNN Task E (new split)",
        priority="high",
        tags=["white_check_mark"],
    )

    # Save & plot loss curves
    import json, matplotlib.pyplot as plt
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f, indent=4)

    plt.figure(figsize=(10, 6))
    epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses,   label="Val Loss",   marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Faster R-CNN — New Sequence Split")
    plt.legend(); plt.grid(True)
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    wandb.log({"loss_curve": wandb.Image(loss_curve_path)})

    # COCO Evaluation
    print("\n--- Running COCO Evaluation ---")
    model.eval()
    results_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            if batch is None:
                continue
            images, targets = batch
            images  = list(img.to(device) for img in images)
            outputs = model(images)
            for output, target in zip(outputs, targets):
                img_id = target["image_id"].item()
                for score, label, bbox in zip(output["scores"], output["labels"], output["boxes"]):
                    label_id = label.item()
                    if label_id in RCNN_TO_COCO_ID:
                        x1, y1, x2, y2 = bbox.tolist()
                        results_list.append({
                            "image_id": img_id,
                            "category_id": RCNN_TO_COCO_ID[label_id],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score.item()
                        })

    if results_list:
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR)
        metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        wandb.log(metrics)
        wandb.run.summary.update(metrics)
    else:
        print("Warning: No detections found.")

    wandb.finish()


if __name__ == "__main__":
    train()
