import os
import random
import sys
import numpy as np
import torch
import albumentations as A
import cv2
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, COCO_CLASSES, VAL_SEQS, TRAIN_SEQS, COCO_TO_RCNN_ID, RCNN_TO_COCO_ID
from utils.notify import notify

SEED = 42
DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_e_def/"

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4

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

def train():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="kitti-mots-rcnn-final",
        name="faster-rcnn-finetuning-def-aug-new-bsval",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "seed": SEED,
            "model": "faster_rcnn_resnet50_fpn",
            "dataset": "KITTI-MOTS",
            "num_classes": 2,
        }
    )

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    num_classes = 2 + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            p=0.2
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))

    train_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, TRAIN_SEQS, transform=train_transforms)
    val_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, VAL_SEQS, transform=None)

    print(f"Train dataset size: {len(train_dataset)}")
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
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    notify(
        f"Training started\nlr={LEARNING_RATE}  bs={BATCH_SIZE}  epochs={NUM_EPOCHS}\n"
        f"train={len(train_dataset)} imgs  val={len(val_dataset)} imgs",
        title="R-CNN Task E",
        tags=["rocket"],
    )

    print("\n--- Starting Fine-Tuning ---")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
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

            batch_loss = losses.item()
            epoch_loss += batch_loss
            global_step += 1

            wandb.log({
                "batch/loss_total":      batch_loss,
                "batch/loss_classifier": loss_dict.get("loss_classifier", torch.tensor(0.0)).item(),
                "batch/loss_box_reg":    loss_dict.get("loss_box_reg",    torch.tensor(0.0)).item(),
                "batch/loss_objectness": loss_dict.get("loss_objectness", torch.tensor(0.0)).item(),
                "batch/loss_rpn_box_reg":loss_dict.get("loss_rpn_box_reg",torch.tensor(0.0)).item(),
            }, step=global_step)

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.train()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
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

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}")

        lr_scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            wandb.run.summary["best_val_loss"] = best_val_loss
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
            notify(
                f"Epoch {epoch+1}/{NUM_EPOCHS} — new best!\ntrain={avg_train_loss:.4f}  val={avg_val_loss:.4f}",
                title="R-CNN Task E",
                tags=["star"],
            )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    print(f"Training complete. Models saved to {OUTPUT_DIR}")
    notify(
        f"Training complete!\nbest_val_loss={best_val_loss:.4f}\nResults: {OUTPUT_DIR}",
        title="R-CNN Task E",
        priority="high",
        tags=["white_check_mark"],
    )

    import json
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, NUM_EPOCHS+1), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN Training Progress")
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    print(f"Loss curve saved to {loss_curve_path}")

    wandb.log({"loss_curve": wandb.Image(loss_curve_path)})

    print("\n--- Running Evaluation on Validation Split ---")
    model.eval()
    results_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            if batch is None:
                continue
            images, targets = batch
            images = list(image.to(device) for image in images)

            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_id = target["image_id"].item()

                for score, label, bbox in zip(output["scores"], output["labels"], output["boxes"]):
                    label_id = label.item()

                    if label_id in RCNN_TO_COCO_ID:
                        coco_label = RCNN_TO_COCO_ID[label_id]
                        x1, y1, x2, y2 = bbox.tolist()
                        coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                        results_list.append({
                            "image_id": img_id,
                            "category_id": coco_label,
                            "bbox": coco_bbox,
                            "score": score.item()
                        })

    if results_list:
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR)
        print(f"Metrics saved to {OUTPUT_DIR}/evaluation_metrics.json")

        import json
        metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            wandb.log(metrics)
            wandb.run.summary.update(metrics)
    else:
        print("Warning: No detections found on validation set!")

    wandb.finish()

if __name__ == "__main__":
    train()
