import sys
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import draw_bboxes, filter_results

DATASET_PATH = "/ghome/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_DIR = "./DeTR/Results_DETR/task_c/"
N = 21  # Number of images to process

def run_inference():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model to Device
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.to(device) # Move model to GPU
    model.eval()
    
    COCO_LABELS = model.config.id2label

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
        valid_boxes, valid_labels, valid_scores = filter_results(results["scores"], results["labels"], results["boxes"])

        # Draw bounding boxes on the image
        image = draw_bboxes(image, valid_boxes, valid_labels, valid_scores, COCO_LABELS, threshold=0.5, box_type="pred")
        
        # Save results
        save_path = os.path.join(OUTPUT_DIR, f"output_detr_{i}.png")
        image.save(save_path)
        print(f"Processed image {i} -> Saved to {save_path}")

    print(f"\nFinished! All images are in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()