import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from peft import PeftModel, PeftConfig

# Import your custom utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, VAL_SEQS, COCO_CLASSES, DETR_TO_COCO_ID
from utils import KittiMotsDataset

# --- Configuration ---
DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
ADAPTER_PATH = "./DeTR/Results_DETR/task_e/final_lora_adapter"
OUTPUT_DIR = "./DeTR/Results_DETR/task_e/eval_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_evaluation():
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load the Adapter Config first to get the base model name
    peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
    
    # 2. Recreate your custom DETR config (Crucial step!)
    # This must match exactly what you used during training
    id2label = {1: "person", 3: "car"}
    label2id = {"person": 1, "car": 3}
    
    config = DetrConfig.from_pretrained(peft_config.base_model_name_or_path)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id
    
    # 3. Load the Base Model using that specific config
    # ignore_mismatched_sizes=True is still needed here
    base_model = DetrForObjectDetection.from_pretrained(
        peft_config.base_model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # 4. NOW load the adapters
    # The shapes will now match: [3, 256] from checkpoint -> [3, 256] in model
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    processor = DetrImageProcessor.from_pretrained(peft_config.base_model_name_or_path)
    val_dataset = KittiMotsDataset(
        DATASET_PATH, 
        ANNOTATION_FILE, 
        processor, 
        VAL_SEQS, 
        transform=None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        collate_fn=collate_fn, 
        shuffle=False, 
        num_workers=2
    )

    # 3. Inference Loop
    results_list = []
    print(f"Running inference on {len(val_dataset)} images...")

    with torch.no_grad():
        for batch in tqdm(val_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = batch["labels"]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Get original image size for scaling
            img_id = labels[0]["image_id"].item()
            img_info = val_dataset.coco.loadImgs(img_id)[0]
            target_sizes = torch.tensor([[img_info['height'], img_info['width']]]).to(device)

            # Threshold=0.0 is mandatory for a valid mAP curve
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.0
            )[0]

            for score, label, bbox in zip(results["scores"], results["labels"], results["boxes"]):
                label_id = label.item()
                if label_id in DETR_TO_COCO_ID:
                    coco_label = DETR_TO_COCO_ID[label_id]
                    x1, y1, x2, y2 = bbox.tolist()
                    results_list.append({
                        "image_id": img_id,
                        "category_id": coco_label,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], # [x, y, w, h]
                        "score": score.item()
                    })

    # 4. Final COCO Evaluation
    if results_list:
        print("\n--- Final mAP Results ---")
        # CRITICAL: Pass val_dataset.ids to fix the "Denominator Trap"
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR)
    else:
        print("Error: No detections generated.")

if __name__ == "__main__":
    run_evaluation()