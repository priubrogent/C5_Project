import sys
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torchvision.transforms.functional as F
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import draw_bboxes, filter_results

DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_c/"
N = 21  # Number of images to process

def run_inference():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model to Device
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()

    # Convert COCO_LABELS list to dictionary {class_id: class_name}
    categories = weights.meta["categories"]
    COCO_LABELS = {i: categories[i] for i in range(len(categories))}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = []
    # Loop 0 to N-1 to find folders and images
    for i in range(N):
        # Automatically pad to 4 digits (0000, 0001...)
        folder = Path(DATASET_PATH) / f"{i:04d}"

        if folder.exists():
            # Get all files, sort them alphabetically, and take the first
            files = sorted(list(folder.glob("*.png")))
            if files:
                images.append(str(files[0]))

    for img_path, i in zip(images, range(len(images))):
        # Preprocess
        image = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(image).to(device)

        # Inference
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        # Post-process - Faster R-CNN returns boxes in [x1, y1, x2, y2] format
        results = {
            "scores": outputs["scores"],
            "labels": outputs["labels"],
            "boxes": outputs["boxes"]
        }

        # Filter results to only include valid classes and prepare for drawing
        valid_boxes, valid_labels, valid_scores = filter_results(results["scores"], results["labels"], results["boxes"])

        # Draw bounding boxes on the image
        image = draw_bboxes(image, valid_boxes, valid_labels, valid_scores, COCO_LABELS, threshold=0.5, box_type="pred")

        # Save results
        save_path = os.path.join(OUTPUT_DIR, f"output_rcnn_{i}.png")
        image.save(save_path)
        print(f"Processed image {i} -> Saved to {save_path}")

    print(f"\nFinished! All images are in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
