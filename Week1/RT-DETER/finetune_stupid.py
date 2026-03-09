import os
import sys
import random
import numpy as np
import torch
import albumentations as A
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import coco_evaluation, plot_loss, TRAIN_SEQS, VAL_SEQS, DETR_TO_COCO_ID
from utils.KittiMotsDataset import KittiMotsDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
DATASET_PATH   = "/home/msiau/data/tmp/amarcos/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "kitti_mots_to_coco_gt.json")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune_stupid")
CHECKPOINT     = "PekingU/rtdetr_r101vd"

# Hyperparameters - ULTRA CONSERVATIVE
# Strategy: Keep 80-class model, minimal adaptation to KITTI
NUM_EPOCHS    = 10      # Short training
BATCH_SIZE    = 16
LEARNING_RATE = 1e-5    # VERY low to avoid disturbing COCO weights
WEIGHT_DECAY  = 1e-4
WARMUP_RATIO  = 0.05    # Minimal warmup
LR_SCHEDULER  = "cosine"
OPTIMIZER     = "adamw_torch_fused"

# LoRA - MINIMAL capacity (just for slight adaptation)
LORA_R     = 8      # Small rank
LORA_ALPHA = 16     # Small alpha

# COCO class IDs we care about (0-indexed in model)
COCO_PERSON_IDX = 0  # person
COCO_CAR_IDX = 2     # car

# For mapping back to KITTI COCO IDs during evaluation
# DETR_TO_COCO_ID maps model output indices to COCO category IDs
# We'll filter COCO predictions: person (model idx 0 → COCO ID 1), car (model idx 2 → COCO ID 3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_transforms():
    """Augmentations matching DeTR's configuration."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.5,
                rotate_limit=0,
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def collate_fn(batch):
    """RT-DETR uses fixed-size inputs."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Custom Dataset that keeps COCO class indices
# ---------------------------------------------------------------------------
class KittiMotsDatasetKeepCOCOClasses(KittiMotsDataset):
    """
    Modified dataset that keeps COCO class indices (0 for person, 2 for car)
    instead of remapping to 0, 1.

    This allows us to train the 80-class model directly without any head modification.
    """
    def __getitem__(self, idx):
        # Get the base item from parent class
        # Returns: {"pixel_values": tensor, "labels": dict}
        item = super().__getitem__(idx)

        # The parent class maps: COCO person (1) → 0, COCO car (3) → 1
        # We need to reverse this to keep COCO indices: 0 → 0 (person), 1 → 2 (car)

        # Remap labels back to COCO indices
        labels = item['labels']
        if 'class_labels' in labels and len(labels['class_labels']) > 0:
            remapped_labels = []
            for label in labels['class_labels']:
                if label.item() == 0:  # KITTI person → COCO person (idx 0)
                    remapped_labels.append(0)
                elif label.item() == 1:  # KITTI car → COCO car (idx 2)
                    remapped_labels.append(2)
                else:
                    remapped_labels.append(label.item())  # shouldn't happen

            labels['class_labels'] = torch.tensor(remapped_labels, dtype=torch.int64)

        return item


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- wandb ---
    wandb.init(
        project="kitti-mots-rtdetr-finetuning",
        entity="veridas",
        name=f"rtdetr-stupid-keep80classes",
        config={
            "checkpoint": CHECKPOINT,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lr_scheduler": LR_SCHEDULER,
            "strategy": "Keep 80-class COCO model, train only on person/car, ultra-conservative"
        },
    )

    # --- Model & processor ---
    print("\n=== Loading COCO pretrained model (80 classes) ===")
    print("Strategy: Keep all 80 classes, train only with person/car data")

    # NO CONFIG MODIFICATION - keep 80 classes as-is
    model = RTDetrForObjectDetection.from_pretrained(CHECKPOINT)
    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)

    print("✓ Loaded 80-class COCO model (no head modification)")

    # --- LoRA ---
    # Ultra-conservative: Small rank, attention only
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Attention only
        lora_dropout=0.1,
        bias="none",
    )

    print("\n=== Applying LoRA ===")
    print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA} (minimal adaptation)")
    print(f"Target modules: {lora_config.target_modules}")

    model = get_peft_model(model, lora_config)

    # Unfreeze detection heads (they'll only get gradients from person/car though)
    _HEAD_KEYWORDS = ("class_embed", "enc_score_head", "denoising_class_embed", "bbox_embed")

    print("\n=== Unfreezing Detection Heads ===")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in _HEAD_KEYWORDS):
            param.requires_grad = True
            unfrozen_count += 1
            if unfrozen_count <= 10:
                print(f"✓ Unfrozen: {name}")
    print(f"... (showing first 10)")
    print(f"Total detection head parameters unfrozen: {unfrozen_count}\n")

    # Verification
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)\n")

    model.to(device)

    # --- Datasets ---
    # Use modified dataset that keeps COCO class indices (0=person, 2=car)
    train_transforms = build_train_transforms()

    train_dataset = KittiMotsDatasetKeepCOCOClasses(
        DATASET_PATH, ANNOTATION_FILE, processor, TRAIN_SEQS,
        transform=train_transforms,
    )
    val_dataset = KittiMotsDatasetKeepCOCOClasses(
        DATASET_PATH, ANNOTATION_FILE, processor, VAL_SEQS,
        transform=None,
    )
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")
    print("Dataset configured to use COCO class indices (person=0, car=2)\n")

    # --- HuggingFace Trainer ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim=OPTIMIZER,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        save_total_limit=2,
        dataloader_num_workers=4,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.005)
        ],
    )

    print("\n" + "="*70)
    print("STARTING 'STUPID' FINE-TUNING (Keep 80 classes, train on person/car)")
    print("="*70)
    print(f"Strategy: No head modification, ultra-conservative adaptation")
    print(f"Training: {NUM_EPOCHS} epochs, LR={LEARNING_RATE} (very low)")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA} (minimal)")
    print(f"Model: 80-class COCO model (unchanged)")
    print(f"Data: Only person/car annotations (classes 0, 2)")
    print(f"Target: Maintain or slightly improve AP@50 = 0.918")
    print("="*70 + "\n")

    trainer.train()

    # Save LoRA adapters
    adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"Adapters saved to: {adapter_path}")

    # --- COCO Evaluation on validation split ---
    print("\n--- Running COCO Evaluation on Validation Split ---")
    model.eval()
    results_list = []

    val_loader = DataLoader(
        val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]

            outputs = model(pixel_values=pixel_values)

            img_id = labels[0]["image_id"].item()
            img_info = val_dataset.coco.loadImgs(img_id)[0]
            target_sizes = torch.tensor([[img_info["height"], img_info["width"]]]).to(device)

            # Post-process with threshold=0 for full PR curve
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]

            # Filter to only person (model idx 0) and car (model idx 2)
            # Map back to KITTI COCO IDs: person (0 → 1), car (2 → 3)
            for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
                label_id = label.item()

                # Only keep person and car predictions
                if label_id == COCO_PERSON_IDX:  # person
                    coco_label = 1  # KITTI COCO person ID
                elif label_id == COCO_CAR_IDX:  # car
                    coco_label = 3  # KITTI COCO car ID
                else:
                    continue  # Skip other classes

                x1, y1, x2, y2 = bbox.tolist()
                results_list.append({
                    "image_id": img_id,
                    "category_id": coco_label,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score.item(),
                })

    if results_list:
        metrics = coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR, save=True)
        plot_loss(trainer, OUTPUT_DIR, save=True)

        # Log to wandb
        if metrics:
            wandb.log(metrics)

            # Print comparison
            print("\n" + "="*70)
            print("RESULTS COMPARISON")
            print("="*70)
            print(f"Pretrained (baseline): AP@50 = 0.918, AP@50:95 = 0.667")
            print(f"'Stupid' (this run):   AP@50 = {metrics.get('AP@0.5', 'N/A'):.3f}, AP@50:95 = {metrics.get('AP@0.5:0.95', 'N/A'):.3f}")

            ap50 = metrics.get('AP@0.5', 0)
            if ap50 > 0.918:
                print("🎉 SUCCESS! Slight improvement over pretrained!")
                print("   (This 'stupid' approach actually worked!)")
            elif ap50 > 0.90:
                print("✓ Good! Maintained pretrained performance")
                print("   (Conservative approach prevented degradation)")
            elif ap50 > 0.85:
                print("⚠ Slight degradation but not catastrophic")
            else:
                print("❌ Performance degraded - use pretrained model")
            print("="*70)
    else:
        print("Warning: no detections produced on the validation set.")

    wandb.finish()


if __name__ == "__main__":
    train()
