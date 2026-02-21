import os
import random
import sys
import numpy as np
import torch
import albumentations as A
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import (
    DetrImageProcessor, 
    DetrForObjectDetection, 
    TrainingArguments,
    DetrConfig,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation,  plot_loss, COCO_CLASSES, VAL_SEQS, TRAIN_SEQS, COCO_TO_DETR_ID, DETR_TO_COCO_ID

# --- Configuration ---
SEED = 42
DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./DeTR/Results_DETR/task_e/"

# --- Hyperparameters ---
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
LR_SCHEDULER = "cosine"
OPTIMIZER = "adamw_torch_fused"

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
    def __init__(self, img_folder, ann_file, processor, sequence_ids, transform=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.transform = transform

        self.ids = [
            idx for idx in self.ids 
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]
        
    def __init__(self, img_folder, ann_file, processor, sequence_ids, transform=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.transform = transform

        self.ids = [
            idx for idx in self.ids 
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]

        # We only keep images and annotations belonging to the selected sequences
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
        # Use cv2 because Albumentations expects numpy arrays
        image = np.array(self.coco.loadImgs(img_id)[0]) # Placeholder logic, use your actual loader
        image = cv2.imread(os.path.join(self.root, img_metadata['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        
        # 2. Extract boxes and labels for Albumentations
        bboxes = []
        class_labels = []
        for ann in target:
            cat_id = ann['category_id']
            if cat_id in COCO_CLASSES and ann.get('iscrowd', 0) == 0:
                bboxes.append(ann['bbox'])
                class_labels.append(COCO_TO_DETR_ID[cat_id])

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

        # 4. Format for the DETR Processor
        # The processor expects the annotations in COCO dict format
        new_target = []
        for box, label in zip(bboxes, class_labels):
            new_target.append({
                "category_id": label,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0
            })

        encoding = self.processor(
            images=image, 
            annotations={'image_id': img_id, 'annotations': new_target}, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0), 
            "labels": encoding["labels"][0]
        }
    
    
def train():
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
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", 
        config=config, 
        ignore_mismatched_sizes=True
    )
    
    # Apply LoRA (Low-Rank Adaptation)
    # We target the attention projections (q_proj, v_proj)
    # We MUST save the classifier heads (modules_to_save) to adapt to KITTI classes
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16,
        target_modules=[
            "q_proj", "v_proj",         # Attention projections in the transformer layers
            "conv1", "conv2", "conv3"   # These are the 3 conv layers in the backbone
        ],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["class_labels_classifier", "bbox_predictor"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # Assuming Class 0 = Pedestrian, Class 1 = Car, Class 2 = Background
    # We give Pedestrians more 'importance' because they are unbalanced
    class_weights = torch.tensor([2.35, 1.0, 0.1]).to(device)
    model.criterion.weight_dict["loss_ce"] = class_weights # Update the criterion weights
    
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
        min_visibility=0.3, # Drops boxes that get too cropped
    ))

    # Initialize Datasets
    train_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, processor, TRAIN_SEQS, transform=train_transforms)
    val_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, processor, VAL_SEQS,transform=None)

    print(f"Number of images the DataLoader sees: {len(val_dataset.ids)}")
    print(f"Number of images in the COCO object: {len(val_dataset.coco.getImgIds())}")

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
    adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapter")
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
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR, imgIds=val_dataset.ids)
        plot_loss(trainer, OUTPUT_DIR)
        print(f"Metrics saved to {OUTPUT_DIR}/evaluation_metrics.json")
    else:
        print("Warning: No detections found on validation set!")


if __name__ == "__main__":
    train()