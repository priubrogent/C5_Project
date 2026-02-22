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

# --- Configuration ---
SEED = 42
DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_e/"

# --- Hyperparameters ---
NUM_EPOCHS = 10
BATCH_SIZE = 4  # Faster R-CNN is memory-intensive, use smaller batch size
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
WARMUP_STEPS = 500
NUM_WORKERS = 4

def set_seed(seed):
    """
    Fix random seeds for reproducibility.
    """
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

        # Filter to only include images from the selected sequences
        self.ids = [
            idx for idx in self.ids
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]

        # Keep only images and annotations belonging to the selected sequences
        val_img_ids_set = set(self.ids)

        self.coco.dataset['images'] = [
            img for img in self.coco.dataset['images'] if img['id'] in val_img_ids_set
        ]
        self.coco.dataset['annotations'] = [
            ann for ann in self.coco.dataset['annotations'] if ann['image_id'] in val_img_ids_set
        ]

        # REBUILD the index
        self.coco.createIndex()

    def __getitem__(self, idx):
        # 1. Load image and raw annotations
        img_id = self.ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]

        # Load image using cv2 for Albumentations
        image = cv2.imread(os.path.join(self.root, img_metadata['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        # 2. Extract boxes and labels for Albumentations
        bboxes = []
        class_labels = []
        for ann in target:
            cat_id = ann['category_id']
            # Only include valid classes (person=1, car=3) and non-crowd annotations
            if cat_id in COCO_CLASSES and ann.get('iscrowd', 0) == 0:
                bboxes.append(ann['bbox'])
                class_labels.append(COCO_TO_RCNN_ID[cat_id])  # Map COCO labels to R-CNN labels

        # 3. Apply Albumentations
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        # 4. Convert to PyTorch format for Faster R-CNN
        # Convert image to tensor
        image = F.to_tensor(image)

        # Convert bboxes from [x, y, w, h] to [x1, y1, x2, y2]
        boxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])

        # Create target dictionary
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(class_labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        return image, target

def collate_fn(batch):
    """
    Custom collator to handle variable number of objects per image.
    """
    return tuple(zip(*batch))

def train():
    # Set seeds for reproducibility
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    wandb.init(
        project="kitti-mots-rcnn",
        name="faster-rcnn-finetuning",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "num_workers": NUM_WORKERS,
            "seed": SEED,
            "model": "faster_rcnn_resnet50_fpn",
            "dataset": "KITTI-MOTS",
            "num_classes": 2,
        }
    )

    # Load pre-trained Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classifier head to match our number of classes
    # We have 2 classes (person, car) + 1 background class
    num_classes = 2 + 1  # 2 foreground classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    # "Safe" augmentations for KITTI-MOTS
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

    # Initialize Datasets
    train_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, TRAIN_SEQS, transform=train_transforms)
    val_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, VAL_SEQS, transform=None)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training loop
    print("\n--- Starting Fine-Tuning ---")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.train()  # Keep in train mode to get losses
        val_loss = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Update learning rate
        lr_scheduler.step()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            wandb.run.summary["best_val_loss"] = best_val_loss
            print(f"Saved best model with val loss: {best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    print(f"Training complete. Models saved to {OUTPUT_DIR}")

    # Save training history
    import json
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Plot loss curves
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

    # Log loss curve to wandb
    wandb.log({"loss_curve": wandb.Image(loss_curve_path)})

    # Run Evaluation on Validation Split
    print("\n--- Running Evaluation on Validation Split ---")
    model.eval()
    results_list = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Inference"):
            images = list(image.to(device) for image in images)

            # Forward pass
            outputs = model(images)

            # Process outputs
            for output, target in zip(outputs, targets):
                img_id = target["image_id"].item()

                for score, label, bbox in zip(output["scores"], output["labels"], output["boxes"]):
                    label_id = label.item()

                    # Filter for valid classes (R-CNN labels: 1=person, 2=car)
                    if label_id in RCNN_TO_COCO_ID:
                        # Map R-CNN labels back to COCO labels
                        coco_label = RCNN_TO_COCO_ID[label_id]
                        x1, y1, x2, y2 = bbox.tolist()
                        coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                        results_list.append({
                            "image_id": img_id,
                            "category_id": coco_label,
                            "bbox": coco_bbox,
                            "score": score.item()
                        })

    # Calculate and Save Metrics using utils.py
    if results_list:
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR)
        print(f"Metrics saved to {OUTPUT_DIR}/evaluation_metrics.json")

        # Log evaluation metrics to wandb
        import json
        metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            wandb.log(metrics)
            wandb.run.summary.update(metrics)
    else:
        print("Warning: No detections found on validation set!")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()
