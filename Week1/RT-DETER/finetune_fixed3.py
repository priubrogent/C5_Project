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
# DATASET_PATH   = "/home/msiau/data/tmp/amarcos/KITTI-MOTS/training/image_02"
DATASET_PATH   = "/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "..", "kitti_mots_to_coco_gt.json")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune_fixed3")
CHECKPOINT     = "PekingU/rtdetr_r101vd"

# Hyperparameters - CONSERVATIVE APPROACH
# Strategy: Smart weight initialization + careful fine-tuning
NUM_EPOCHS    = 15      # Medium training length
BATCH_SIZE    = 16
LEARNING_RATE = 5e-5    # LOWER than before (was 1e-4) to preserve COCO knowledge
WEIGHT_DECAY  = 1e-4
WARMUP_RATIO  = 0.1
LR_SCHEDULER  = "linear"
OPTIMIZER     = "adamw_torch_fused"

# LoRA - Moderate capacity
LORA_R     = 16
LORA_ALPHA = 32

# RT-DETR class mapping (2 classes: person and car)
ID2LABEL = {0: "person", 1: "car"}
LABEL2ID = {"person": 0, "car": 1}

# COCO class indices for person and car (0-indexed in model weights)
COCO_PERSON_IDX = 0  # COCO class 1 (person) is at index 0
COCO_CAR_IDX = 2     # COCO class 3 (car) is at index 2


# ---------------------------------------------------------------------------
# Smart Weight Initialization
# ---------------------------------------------------------------------------
def initialize_from_coco_weights(model_2class, model_coco):
    """
    Initialize the 2-class detection heads with COCO person/car weights.

    This is the KEY to preserving COCO knowledge:
    - COCO trained on person (class 0) and car (class 2)
    - We copy these weights to our 2-class model
    - This gives a warm start instead of random initialization
    """
    print("\n=== Initializing 2-class heads from COCO person/car weights ===")

    # RT-DETR has 6 decoder layers, each with class_embed and bbox_embed
    num_decoder_layers = len(model_coco.model.decoder.class_embed)

    for layer_idx in range(num_decoder_layers):
        # --- Class Embedding ---
        # Copy person (COCO idx 0) → KITTI person (idx 0)
        # Copy car (COCO idx 2) → KITTI car (idx 1)
        coco_class_weight = model_coco.model.decoder.class_embed[layer_idx].weight.data
        coco_class_bias = model_coco.model.decoder.class_embed[layer_idx].bias.data

        model_2class.model.decoder.class_embed[layer_idx].weight.data[0] = coco_class_weight[COCO_PERSON_IDX]
        model_2class.model.decoder.class_embed[layer_idx].weight.data[1] = coco_class_weight[COCO_CAR_IDX]
        model_2class.model.decoder.class_embed[layer_idx].bias.data[0] = coco_class_bias[COCO_PERSON_IDX]
        model_2class.model.decoder.class_embed[layer_idx].bias.data[1] = coco_class_bias[COCO_CAR_IDX]

        print(f"✓ Layer {layer_idx} class_embed: Copied COCO person/car weights")

        # --- Bbox Embedding ---
        # Bbox predictor is class-agnostic (same for all classes), so we just copy all weights
        # bbox_embed is an MLP with 3 layers
        coco_bbox_mlp = model_coco.model.decoder.bbox_embed[layer_idx]
        kitti_bbox_mlp = model_2class.model.decoder.bbox_embed[layer_idx]

        for mlp_layer_idx in range(len(coco_bbox_mlp.layers)):
            kitti_bbox_mlp.layers[mlp_layer_idx].weight.data = coco_bbox_mlp.layers[mlp_layer_idx].weight.data.clone()
            kitti_bbox_mlp.layers[mlp_layer_idx].bias.data = coco_bbox_mlp.layers[mlp_layer_idx].bias.data.clone()

        print(f"✓ Layer {layer_idx} bbox_embed: Copied COCO bbox MLP weights")

    # Also initialize denoising class embed (used during training)
    # COCO has 81 classes (80 + background), we have 3 (2 + background)
    if hasattr(model_2class.model, 'denoising_class_embed') and hasattr(model_coco.model, 'denoising_class_embed'):
        model_2class.model.denoising_class_embed.weight.data[0] = model_coco.model.denoising_class_embed.weight.data[COCO_PERSON_IDX]
        model_2class.model.denoising_class_embed.weight.data[1] = model_coco.model.denoising_class_embed.weight.data[COCO_CAR_IDX]
        # Keep background (last class) as random initialization
        print(f"✓ denoising_class_embed: Copied COCO person/car weights")

    print("=== Weight initialization complete! ===\n")
    return model_2class


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
    """RT-DETR uses fixed-size inputs (resized by the processor)."""
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
    wandb.init(
        project="kitti-mots-rtdetr-finetuning",
        entity="veridas",
        name=f"rtdetr-lora-r{LORA_R}-fixed3-smart-init",
        config={
            "checkpoint": CHECKPOINT,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lr_scheduler": LR_SCHEDULER,
            "strategy": "Smart weight initialization from COCO person/car classes"
        },
    )

    # --- Load COCO pretrained model FIRST (for weight extraction) ---
    print("\n=== Loading COCO pretrained model ===")
    model_coco = RTDetrForObjectDetection.from_pretrained(CHECKPOINT)
    print("✓ COCO model loaded (80 classes)")

    # --- Create 2-class model config ---
    model_config = RTDetrConfig.from_pretrained(CHECKPOINT)
    model_config.num_labels = len(ID2LABEL)
    model_config.id2label   = ID2LABEL
    model_config.label2id   = LABEL2ID

    # --- Load 2-class model (initially with random heads) ---
    print("\n=== Loading 2-class model ===")
    model = RTDetrForObjectDetection.from_pretrained(
        CHECKPOINT,
        config=model_config,
        ignore_mismatched_sizes=True,   # This creates new 2-class heads (random init)
    )
    print("✓ 2-class model loaded (random head initialization)")

    # --- Initialize 2-class heads with COCO person/car weights ---
    model = initialize_from_coco_weights(model, model_coco)

    # Free memory - don't need COCO model anymore
    del model_coco
    torch.cuda.empty_cache()

    # --- Model & processor ---
    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)

    # --- LoRA ---
    # Conservative approach: Only target attention layers, NOT backbone convolutions
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # Attention only
        lora_dropout=0.1,
        bias="none",
    )

    print("\n=== Applying LoRA ===")
    print(f"LoRA config: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"Target modules: {lora_config.target_modules}")

    model = get_peft_model(model, lora_config)

    # Manually unfreeze detection heads AFTER PEFT wrapping
    _HEAD_KEYWORDS = ("class_embed", "enc_score_head", "denoising_class_embed", "bbox_embed")

    print("\n=== Unfreezing Detection Heads ===")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in _HEAD_KEYWORDS):
            param.requires_grad = True
            unfrozen_count += 1
            # Only print first few to avoid spam
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
    print("STARTING RT-DETR FINE-TUNING WITH SMART WEIGHT INITIALIZATION")
    print("="*70)
    print(f"Strategy: Initialize 2-class heads with COCO person/car weights")
    print(f"Training: {NUM_EPOCHS} epochs, LR={LEARNING_RATE}, {LR_SCHEDULER} scheduler")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}, attention layers only")
    print(f"Target: Beat pretrained AP@50 = 0.918")
    print("="*70 + "\n")

    trainer.train()

    # Save LoRA adapters
    adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapter")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"Adapters saved to: {adapter_path}")

    # Save detection head weights separately — PEFT only stores LoRA matrices,
    # so class_embed / bbox_embed / enc_score_head / denoising_class_embed must
    # be persisted explicitly so they can be restored at inference time.
    _HEAD_KEYWORDS = ("class_embed", "enc_score_head", "denoising_class_embed", "bbox_embed")
    head_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if any(kw in name for kw in _HEAD_KEYWORDS)
    }
    torch.save(head_state, os.path.join(adapter_path, "detection_heads.pt"))
    print(f"Detection head weights saved to: {adapter_path}/detection_heads.pt")

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
            print(f"Fixed v3 (this run):   AP@50 = {metrics.get('AP@0.5', 'N/A'):.3f}, AP@50:95 = {metrics.get('AP@0.5:0.95', 'N/A'):.3f}")
            if metrics.get('AP@0.5', 0) > 0.918:
                print("🎉 SUCCESS! Fine-tuning improved over pretrained model!")
            elif metrics.get('AP@0.5', 0) > 0.90:
                print("✓ Good result, close to pretrained performance")
            else:
                print("⚠ Performance degraded - consider using pretrained model")
            print("="*70)
    else:
        print("Warning: no detections produced on the validation set.")

    wandb.finish()


if __name__ == "__main__":
    train()
