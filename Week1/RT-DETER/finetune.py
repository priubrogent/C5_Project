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
    RTDetrConfig,
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
DATASET_PATH   = "/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "kitti_mots_to_coco_gt.json")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune")
CHECKPOINT     = "PekingU/rtdetr_r101vd"   # r50vd also works and is lighter

# Hyperparameters
NUM_EPOCHS    = 10
BATCH_SIZE    = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_RATIO  = 0.1
LR_SCHEDULER  = "cosine"
OPTIMIZER     = "adamw_torch_fused"

# LoRA
LORA_R     = 16
LORA_ALPHA = 32

# RT-DETR class mapping (2 classes: person and car)
# The dataset converts COCO IDs (1, 3) → model indices (0, 1) via COCO_TO_DETR_ID.
# id2label is stored in the config for display and post-processing.
ID2LABEL = {1: "person", 3: "car"}
LABEL2ID = {"person": 1, "car": 3}


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
    """Albumentations augmentations that are safe for driving scenes."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=(-0.1, 0.1), scale=(0.7, 1.3), rotate=0, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
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
    """
    RT-DETR uses fixed-size inputs (resized by the processor), so we can
    simply stack pixel_values — no pixel_mask padding needed unlike DETR.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- wandb ---
    """
    wandb.init(
        project="C5-RT-DETR",
        name=f"rtdetr-lora-r{LORA_R}",
        config={
            "checkpoint": CHECKPOINT,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
        },
    )
    """
    
    # --- Model & processor ---
    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)

    model_config = RTDetrConfig.from_pretrained(CHECKPOINT)
    model_config.num_labels = len(ID2LABEL)
    model_config.id2label   = ID2LABEL
    model_config.label2id   = LABEL2ID

    model = RTDetrForObjectDetection.from_pretrained(
        CHECKPOINT,
        config=model_config,
        ignore_mismatched_sizes=True,   # replaces the COCO 80-class head with a 2-class head
    )

    # --- LoRA ---
    # RT-DETR decoder cross-attention uses q_proj / k_proj / v_proj / out_proj.
    # NOTE: we do NOT use modules_to_save here because RT-DETR names its detection
    # heads differently from DETR ('class_embed', 'enc_score_head', etc.) and PEFT
    # would silently ignore unrecognised names, leaving the heads frozen.
    # Instead we manually unfreeze them below.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Unfreeze the RT-DETR detection heads that were re-initialised for 2 classes.
    # These correspond to the MISMATCH keys in the load report and must be learned.
    _HEAD_KEYWORDS = ("class_embed", "enc_score_head", "denoising_class_embed", "bbox_embed")
    for name, param in model.named_parameters():
        if any(kw in name for kw in _HEAD_KEYWORDS):
            param.requires_grad = True

    model.print_trainable_parameters()
    model.to(device)

    # --- Datasets ---
    train_transforms = build_train_transforms()

    train_dataset = KittiMotsDataset(
        DATASET_PATH, ANNOTATION_FILE, processor, TRAIN_SEQS,
        transform=train_transforms,
    )
    val_dataset = KittiMotsDataset(
        DATASET_PATH, ANNOTATION_FILE, processor, VAL_SEQS,
        transform=None,
    )
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

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
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.005)
        ],
    )

    print("\n--- Starting RT-DETR LoRA Fine-Tuning ---")
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

            # RT-DETR does not use pixel_mask
            outputs = model(pixel_values=pixel_values)

            img_id = labels[0]["image_id"].item()
            img_info = val_dataset.coco.loadImgs(img_id)[0]
            target_sizes = torch.tensor([[img_info["height"], img_info["width"]]]).to(device)

            # threshold=0 is required for a full PR curve and correct mAP
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]

            for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
                label_id = label.item()
                if label_id in DETR_TO_COCO_ID:
                    coco_label = DETR_TO_COCO_ID[label_id]
                    x1, y1, x2, y2 = bbox.tolist()
                    results_list.append({
                        "image_id": img_id,
                        "category_id": coco_label,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO [x, y, w, h]
                        "score": score.item(),
                    })

    if results_list:
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR, save=True)
        plot_loss(trainer, OUTPUT_DIR, save=True)
    else:
        print("Warning: no detections produced on the validation set.")

    # wandb.finish()


if __name__ == "__main__":
    train()
