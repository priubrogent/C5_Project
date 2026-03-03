import os
import random
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import cv2
import wandb
from PIL import Image
from io import BytesIO
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from tqdm import tqdm
from datasets import load_dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.notify import notify

SEED = 42
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_f/"

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_SPLIT = 0.2

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

DEART_PERSON_CLASSES = {"person", "nude"}
COCO_PERSON_ID = 1
NUM_CLASSES = 2


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


class DEArtDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self._build_category_id_to_name()

    def _build_category_id_to_name(self):
        self.cat_id_to_name = {}

        for item in self.dataset:
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
        image = np.array(image)

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

        if self.transform and len(bboxes) > 0:
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
            "image_id": torch.tensor([idx]),
            "bboxes_coco": bboxes,
        }

        return image, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None and len(b[1]["boxes"]) > 0]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))


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


def build_coco_gt_from_dataset(dataset, output_path):
    images = []
    annotations = []
    ann_id = 1

    print("Building COCO GT for evaluation...")
    for idx in tqdm(range(len(dataset)), desc="Building GT"):
        _, target = dataset[idx]
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

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}]
    }

    gt_path = os.path.join(output_path, "deart_gt_coco.json")
    with open(gt_path, "w") as f:
        json.dump(coco_dict, f)

    return COCO(gt_path)


def train():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading DEArt dataset from HuggingFace...")
    full_dataset = load_dataset("biglam/european_art", split="train")
    print(f"Total dataset size: {len(full_dataset)}")

    print(f"Splitting dataset: {int((1-VAL_SPLIT)*100)}% train / {int(VAL_SPLIT*100)}% val")
    split_dataset = full_dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_hf = split_dataset["train"]
    val_hf = split_dataset["test"]

    print(f"Train size: {len(train_hf)}, Val size: {len(val_hf)}")

    print("\nLoading Faster R-CNN...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

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

    wandb.init(
        project="deart-rcnn-finetuning",
        name="faster-rcnn-deart-lora",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_workers": NUM_WORKERS,
            "seed": SEED,
            "model": "faster_rcnn_resnet50_fpn",
            "dataset": "DEArt (European Art) - person/nude only",
            "num_classes": NUM_CLASSES,
            "train_size": len(train_hf),
            "val_size": len(val_hf),
            "target_classes": list(DEART_PERSON_CLASSES),
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "trainable_params": trainable_params,
        }
    )

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))

    train_dataset = DEArtDataset(train_hf, transform=train_transforms)
    val_dataset = DEArtDataset(val_hf, transform=None)

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
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mapping_info = {
        "deart_classes": list(DEART_PERSON_CLASSES),
        "coco_target_id": COCO_PERSON_ID,
        "num_classes": NUM_CLASSES,
        "class_names": {0: "background", 1: "person"}
    }
    with open(os.path.join(OUTPUT_DIR, "category_mapping.json"), "w") as f:
        json.dump(mapping_info, f, indent=4)

    notify(
        f"DEArt Training started\nlr={LEARNING_RATE}  bs={BATCH_SIZE}  epochs={NUM_EPOCHS}\n"
        f"train={len(train_dataset)} imgs  val={len(val_dataset)} imgs\n"
        f"classes: person/nude -> COCO person (1)",
        title="R-CNN Task F (DEArt)",
        tags=["rocket"],
    )

    print("\n--- Starting Fine-Tuning on DEArt ---")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training"):
            if batch is None:
                continue

            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_loss = losses.item()
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            wandb.log({
                "batch/loss_total": batch_loss,
                "batch/loss_classifier": loss_dict.get("loss_classifier", torch.tensor(0.0)).item(),
                "batch/loss_box_reg": loss_dict.get("loss_box_reg", torch.tensor(0.0)).item(),
                "batch/loss_objectness": loss_dict.get("loss_objectness", torch.tensor(0.0)).item(),
                "batch/loss_rpn_box_reg": loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item(),
            }, step=global_step)

        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)

        model.train()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation"):
                if batch is None:
                    continue

                images, targets = batch
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                num_val_batches += 1

        avg_val_loss = val_loss / max(num_val_batches, 1)
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
                f"Epoch {epoch+1}/{NUM_EPOCHS} - new best!\ntrain={avg_train_loss:.4f}  val={avg_val_loss:.4f}",
                title="R-CNN Task F (DEArt)",
                tags=["star"],
            )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pth"))
    print(f"Training complete. Models saved to {OUTPUT_DIR}")

    notify(
        f"DEArt Training complete!\nbest_val_loss={best_val_loss:.4f}\nResults: {OUTPUT_DIR}",
        title="R-CNN Task F (DEArt)",
        priority="high",
        tags=["white_check_mark"],
    )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "trainable_params": trainable_params,
    }
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Faster R-CNN DEArt LoRA Fine-Tuning Progress")
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to {loss_curve_path}")

    wandb.log({"loss_curve": wandb.Image(loss_curve_path)})

    print("\n--- Loading Best Model for Evaluation ---")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()

    coco_gt = build_coco_gt_from_dataset(val_dataset, OUTPUT_DIR)

    print("\n--- Running Evaluation on Validation Split ---")
    results_list = []

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Inference"):
            image, target = val_dataset[idx]
            image = image.to(device)
            img_id = idx

            outputs = model([image])[0]

            for score, label, bbox in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
                if label.item() == COCO_PERSON_ID:
                    x1, y1, x2, y2 = bbox.tolist()
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                    results_list.append({
                        "image_id": img_id,
                        "category_id": COCO_PERSON_ID,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    with open(os.path.join(OUTPUT_DIR, "predictions.json"), "w") as f:
        json.dump(results_list, f)
    print(f"Predictions saved to {OUTPUT_DIR}/predictions.json")

    print(f"Ground truth annotations: {len(coco_gt.getAnnIds())}")
    print(f"Total person detections: {len(results_list)}")

    metrics = coco_evaluation_single_class(results_list, coco_gt, OUTPUT_DIR)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY (LoRA Fine-tuned model on DEArt)")
    print("="*50)
    print(f"Dataset: DEArt (European Art) - Validation Split ({int(VAL_SPLIT*100)}%)")
    print(f"Classes evaluated: person, nude -> COCO person")
    print(f"Total images: {len(val_dataset)}")
    print(f"GT annotations: {len(coco_gt.getAnnIds())}")
    print(f"Total detections: {len(results_list)}")
    print("-"*50)
    print(f"mAP@0.50:0.95 = {metrics.get('mAP_0.50_0.95', 'N/A')}")
    print(f"mAP@0.50      = {metrics.get('mAP_0.50', 'N/A')}")
    print(f"mAP@0.75      = {metrics.get('mAP_0.75', 'N/A')}")
    print(f"mAR@100       = {metrics.get('mAR_100', 'N/A')}")
    print("="*50)

    for key, value in metrics.items():
        wandb.run.summary[f"eval/{key}"] = value

    print("\n--- Generating Qualitative Visualizations ---")
    generate_qualitative_results(model, val_hf, device, OUTPUT_DIR, num_samples=10)

    wandb.finish()

    notify(
        f"DEArt LoRA Training complete!\n"
        f"mAP@0.50={metrics.get('mAP_0.50', 'N/A')}\n"
        f"mAP@0.50:0.95={metrics.get('mAP_0.50_0.95', 'N/A')}\n"
        f"Results: {OUTPUT_DIR}",
        title="R-CNN Task F (DEArt LoRA)",
        priority="high",
        tags=["white_check_mark"],
    )


def generate_qualitative_results(model, val_hf, device, output_dir, num_samples=10, score_threshold=0.5):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    viz_dir = os.path.join(output_dir, "qualitative_results")
    os.makedirs(viz_dir, exist_ok=True)

    model.eval()

    random.seed(SEED)
    sample_indices = random.sample(range(len(val_hf)), min(num_samples, len(val_hf)))

    for i, idx in enumerate(tqdm(sample_indices, desc="Generating visualizations")):
        item = val_hf[idx]

        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

        ann_data = json.loads(item['annotations'])
        annotations = ann_data.get('annotations', [])

        local_cat_id_to_name = {}
        for cat in ann_data.get('categories', []):
            local_cat_id_to_name[cat['id']] = cat['name'].lower()

        gt_bboxes = []
        for ann in annotations:
            if ann.get('iscrowd', 0) == 0:
                cat_name = local_cat_id_to_name.get(ann['category_id'], '').lower()
                if cat_name in DEART_PERSON_CLASSES:
                    gt_bboxes.append(ann['bbox'])

        image_tensor = F.to_tensor(image_np).to(device)
        with torch.no_grad():
            outputs = model([image_tensor])[0]

        pred_bboxes = []
        pred_scores = []
        for score, label, bbox in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
            if label.item() == COCO_PERSON_ID and score.item() >= score_threshold:
                x1, y1, x2, y2 = bbox.tolist()
                pred_bboxes.append([x1, y1, x2 - x1, y2 - y1])
                pred_scores.append(score.item())

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(image_np)
        axes[0].set_title(f"Ground Truth ({len(gt_bboxes)} persons)", fontsize=14)
        axes[0].axis('off')
        for bbox in gt_bboxes:
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            axes[0].add_patch(rect)

        axes[1].imshow(image_np)
        axes[1].set_title(f"Predictions ({len(pred_bboxes)} detections, threshold={score_threshold})", fontsize=14)
        axes[1].axis('off')
        for bbox, score in zip(pred_bboxes, pred_scores):
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(x, y - 5, f"{score:.2f}", color='red', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        plt.tight_layout()
        save_path = os.path.join(viz_dir, f"sample_{i+1:02d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved {num_samples} qualitative visualizations to {viz_dir}")

    create_grid_visualization(viz_dir, output_dir, num_samples)


def create_grid_visualization(viz_dir, output_dir, num_samples):
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    images = []
    for i in range(min(num_samples, 6)):
        img_path = os.path.join(viz_dir, f"sample_{i+1:02d}.png")
        if os.path.exists(img_path):
            images.append(PILImage.open(img_path))

    if len(images) == 0:
        return

    n_cols = min(3, len(images))
    n_rows = (len(images) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 4))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')

    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.suptitle("DEArt Detection Results: Ground Truth (Green) vs Predictions (Red)", fontsize=16, y=1.02)
    plt.tight_layout()

    grid_path = os.path.join(output_dir, "qualitative_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grid visualization to {grid_path}")


if __name__ == "__main__":
    train()
