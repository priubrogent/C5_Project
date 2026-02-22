import os
import random
import sys
import numpy as np
import torch
import albumentations as A
import cv2
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DetrImageProcessor, 
    DetrForObjectDetection, 
    TrainingArguments,
    DetrConfig,
    Trainer,
    EarlyStoppingCallback
)
from peft import PeftModel, LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.utils import coco_evaluation, plot_loss, VAL_SEQS, TRAIN_SEQS, DETR_TO_COCO_ID
from utils.KittiMotsDataset import KittiMotsDataset

# --- Configuration ---
SEED = 42
DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./DeTR/Results_DETR/task_e/"
FILE_NAME = "ablation_3_4_6_1"  # Update this for each ablation run
LORA_ADAPTER_DIR = "./DeTR/Results_DETR/task_e/lora_adapter"

# --- Hyperparameters ---
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_torch_fused"
BACKBONE_ABLATION = {
    1: 3, # 3 in total
    2: 4, # 4 in total
    3: 6, # 6 in total
    4: 1  # 3 in total
}
TRANSFORMER_ABLATION = {
    "encoder": 6, # 6 in total
    "decoder": 6  # 6 in total
}

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


def prune_resnet_backbone(model, pruning_dict):
    """
    Updated for Hugging Face DetrForObjectDetection structure.
    pruning_dict: {stage_number: blocks_to_keep}
    """
    # 1. Reach the internal DETR model
    # If using PEFT, we go through get_base_model()
    curr_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    
    # 2. Correct path to the ResNet object in HF Transformers
    # model.model.backbone -> DetrConvEncoder
    # model.model.backbone.backbone -> DetrResnetBackbone
    # model.model.backbone.backbone.model -> The actual ResNet with layer1, layer2, etc.
    try:
        backbone = curr_model.model.backbone.backbone.model
    except AttributeError:
        # Fallback for some versions of the library
        backbone = curr_model.model.backbone.model
        
    print(f"Successfully reached backbone: {type(backbone).__name__}")

    for stage_num, keep_count in pruning_dict.items():
        stage_name = f"layer{stage_num}"
        if not hasattr(backbone, stage_name):
            print(f"Warning: Stage {stage_num} ({stage_name}) not found. Skipping.")
            continue
            
        stage = getattr(backbone, stage_name)
        total_blocks = len(stage)
        
        # Guard: Ensure we keep at least the downsampling block (index 0)
        if keep_count < 1:
            keep_count = 1
            
        if keep_count >= total_blocks:
            continue

        # Replace unwanted blocks with Identity
        for i in range(keep_count, total_blocks):
            stage[i] = nn.Identity()
            
        print(f"Pruned Stage {stage_num}: Reduced from {total_blocks} to {keep_count} active blocks.")
        
def prune_transformer_layers(model, encoder_keep=None, decoder_keep=None):
    """
    Prunes the Transformer encoder and decoder layers.
    encoder_keep: Number of layers to keep in the encoder (default 6)
    decoder_keep: Number of layers to keep in the decoder (default 6)
    """
    curr_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    
    # Path: model.model.encoder.layers and model.model.decoder.layers
    transformer = curr_model.model
    
    # 1. Prune Encoder
    if encoder_keep is not None:
        layers = transformer.encoder.layers
        total = len(layers)
        if encoder_keep < 1: encoder_keep = 1
        if encoder_keep < total:
            for i in range(encoder_keep, total):
                layers[i] = nn.Identity()
            print(f"Pruned Transformer Encoder: Kept {encoder_keep}/{total} layers.")

    # 2. Prune Decoder
    if decoder_keep is not None:
        layers = transformer.decoder.layers
        total = len(layers)
        if decoder_keep < 1: decoder_keep = 1
        if decoder_keep < total:
            for i in range(decoder_keep, total):
                layers[i] = nn.Identity()
            print(f"Pruned Transformer Decoder: Kept {decoder_keep}/{total} layers.")
    
def ablation():
    # Set seeds for reproducibility
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Since the DETR model trained in COCO has 81 classes (including background), 
    # we need to adapt it to our 2 classes (car and pedestrian)
    # Define the mapping
    id2label = {1: "person", 3: "car"}
    label2id = {"person": 1, "car": 3}

    # Load config and update the number of labels
    # Note: DETR adds an extra 'no-object' class automatically
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = len(id2label) # Just 2 classes (person and car)
    config.id2label = id2label
    config.label2id = label2id

    # Prepare Model and Processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    # Load the model with the new head
    base_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", 
        config=config, 
        ignore_mismatched_sizes=True
    )
    
    # Charge the LoRA adapters previously obtained from fine-tuning the whole network
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR, is_trainable=True)
    model.print_trainable_parameters()
    model.to(device)
    
    # Do the structured pruning based on the provided ablation configuration
    prune_resnet_backbone(model, BACKBONE_ABLATION)
    prune_transformer_layers(
        model, 
        encoder_keep=TRANSFORMER_ABLATION["encoder"], 
        decoder_keep=TRANSFORMER_ABLATION["decoder"]
    )
    
    # "Safe" augmentations for KITTI-MOTS
    train_transforms = A.Compose([
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
    ], bbox_params=A.BboxParams(
        format='coco', 
        label_fields=['class_labels'],
        min_visibility=0.3, # Drops boxes that get too cropped
    ))

    # Initialize Datasets
    train_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, processor, TRAIN_SEQS, transform=train_transforms)
    val_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, processor, VAL_SEQS,transform=None)

    #print(f"Number of images the DataLoader sees: {len(val_dataset.ids)}")
    #print(f"Number of images in the COCO object: {len(val_dataset.coco.getImgIds())}")
    
    def collate_fn(batch):
        """
        Custom collator to handle variable number of objects per image.
        """
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad images to the largest size in the batch
        encoding = processor.pad(pixel_values, return_tensors="pt")
        
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels
        }

    # Define Training Arguments
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
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",       
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(), # Use fp16 to accelerate training if on GPU
        seed=SEED,
        data_seed=SEED,
        save_total_limit=2,
        dataloader_num_workers=4
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start Training
    print("\n--- Starting LoRA Fine-Tuning ---")
    trainer.train()
    
    # Save the final LoRA adapters
    adapter_path = os.path.join(OUTPUT_DIR, f"{FILE_NAME}_lora_adapter")
    model.save_pretrained(adapter_path)
    print(f"Training complete. Adapters saved to {adapter_path}")

    # Run Evaluation on Validation Split
    print("\n--- Running Evaluation on Validation Split ---")
    model.eval()
    results_list = []
    
    # Use a DataLoader for efficient inference
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            # Move data to GPU
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = batch["labels"]

            # Forward pass
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Post-process: Get original image size for correct scaling
            # Since batch_size=1, we take the first element
            img_id = labels[0]["image_id"].item()
            img_info = val_dataset.coco.loadImgs(img_id)[0]
            target_sizes = torch.tensor([[img_info['height'], img_info['width']]]).to(device)

            # threshold=0.0 is crucial for mAP calculation (requires full PR curve)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)[0]

            for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
                label_id = label.item()
                
                # Filter strictly for our target classes (Car/Person)
                if label_id in DETR_TO_COCO_ID:
                    coco_label = DETR_TO_COCO_ID[label_id]
                    x1, y1, x2, y2 = bbox.tolist()
                    # Convert [x1, y1, x2, y2] to COCO [x, y, w, h]
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1] 

                    results_list.append({
                        "image_id": img_id,
                        "category_id": coco_label,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    # Calculate and Save Metrics using utils.py
    if results_list:
        # We pass the underlying COCO object from val_dataset to the evaluator
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR, file_name=f"{FILE_NAME}.json")
        plot_loss(trainer, OUTPUT_DIR, file_name=f"{FILE_NAME}.png")
        print(f"Metrics saved to {OUTPUT_DIR}/{FILE_NAME}")
    else:
        print("Warning: No detections found on validation set!")


if __name__ == "__main__":
    ablation()