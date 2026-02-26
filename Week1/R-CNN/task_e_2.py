import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from task_e import KittiMotsDataset, collate_fn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, VAL_SEQS, RCNN_TO_COCO_ID

# --- Configuration ---
DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
ANNOTATION_FILE = "kitti_mots_to_coco_gt.json"
MODEL_PATH = "./R-CNN/Results_RCNN/task_e_new_split/best_model.pth"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_e_new_split/eval_results"


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Load architecture and restore saved weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)  # bg + person + car

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    val_dataset = KittiMotsDataset(DATASET_PATH, ANNOTATION_FILE, VAL_SEQS, transform=None)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Running inference on {len(val_dataset)} images (VAL_SEQS={VAL_SEQS})...")

    results_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            if batch is None:
                continue
            images, targets = batch
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_id = target["image_id"].item()
                for score, label, bbox in zip(output["scores"], output["labels"], output["boxes"]):
                    label_id = label.item()
                    if label_id in RCNN_TO_COCO_ID:
                        coco_label = RCNN_TO_COCO_ID[label_id]
                        x1, y1, x2, y2 = bbox.tolist()
                        results_list.append({
                            "image_id": img_id,
                            "category_id": coco_label,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score.item()
                        })

    if results_list:
        coco_evaluation(results_list, val_dataset.coco, OUTPUT_DIR)
        print(f"Metrics saved to {OUTPUT_DIR}/evaluation_metrics.json")
    else:
        print("Warning: No detections generated.")


if __name__ == "__main__":
    run_evaluation()
