import sys
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import os
from pathlib import Path
import torch.nn as nn
from transformers import (
    DetrImageProcessor, 
    DetrForObjectDetection, 
    DetrConfig,
)
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.utils import draw_bboxes, filter_results

DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_DIR = "./DeTR/Results_DETR/task_e/qualitative/"
N = 21  # Number of images to process
BACKBONE_ABLATION = {
    1: 3, # 3 in total
    2: 4, # 4 in total
    3: 6, # 6 in total
    4: 1  # 3 in total
}
LORA_ADAPTER_DIR = "./DeTR/Results_DETR/task_e/ablation_3_4_6_1_lora_adapter"


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


def run_inference():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Since the DETR model trained in COCO has 81 classes (including background), 
    # we need to adapt it to our 2 classes (car and pedestrian)
    # Define the mapping
    id2label = {0: "person", 1: "car"}
    label2id = {"person": 0, "car": 1}

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
    prune_resnet_backbone(model, BACKBONE_ABLATION)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    COCO_LABELS = model.config.id2label
    print(COCO_LABELS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    images = []
    # Loop 0 to N-1 to find folders and images
    for i in range(N):
        # Automatically pad to 4 digits (0000, 0001...)
        folder = Path(DATASET_PATH) / f"{i:04d}"
        
        if folder.exists():
            # Get all files, sort them alphabetically, and take the first
            files = sorted(list(folder.glob("*.png"))) # Change extension if needed
            if files:
                images.append(str(files[0]))

    for img_path, i in zip(images, range(len(images))):
        # Preprocess
        image = Image.open(img_path).convert("RGB")
        
        # Move inputs to device
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0)[0]
        
        # Filter results to only include valid classes and prepare for drawing
        #valid_boxes, valid_labels, valid_scores = filter_results(results["scores"], results["labels"], results["boxes"])

        # Draw bounding boxes on the image
        image = draw_bboxes(image, results["boxes"], results["labels"], results["scores"], COCO_LABELS, threshold=0.5, box_type="pred")
        
        # Save results
        save_path = os.path.join(OUTPUT_DIR, f"output_detr_{i}.png")
        image.save(save_path)
        print(f"Processed image {i} -> Saved to {save_path}")

    print(f"\nFinished! All images are in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()