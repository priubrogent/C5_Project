"""
Evaluate the trained LoRA model.
"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from tqdm import tqdm
from pycocotools.coco import COCO
import torch.nn as nn
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import coco_evaluation, VAL_SEQS, RCNN_TO_COCO_ID

DATASET_PATH = "/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02"
GT_PATH = "kitti_mots_to_coco_gt.json"
MODEL_PATH = "./R-CNN/Results_RCNN/task_e_lora/best_model.pth"
OUTPUT_DIR = "./R-CNN/Results_RCNN/task_e_lora/"

# LoRA hyperparameters (must match training)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


class LoRALayer(nn.Module):
    """
    LoRA layer for Conv2d.
    """
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Get layer dimensions
        if isinstance(original_layer, nn.Conv2d):
            self.in_features = original_layer.in_channels
            self.out_features = original_layer.out_channels
            kernel_size = original_layer.kernel_size
            stride = original_layer.stride
            padding = original_layer.padding

            self.lora_A = nn.Conv2d(
                self.in_features, rank,
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.lora_B = nn.Conv2d(
                rank, self.out_features,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            )
        elif isinstance(original_layer, nn.Linear):
            self.in_features = original_layer.in_features
            self.out_features = original_layer.out_features

            self.lora_A = nn.Linear(self.in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        else:
            raise ValueError(f"LoRA not supported for {type(original_layer)}")

        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return original_output + lora_output * self.scaling


def apply_lora_to_model(model, rank=8, alpha=16, dropout=0.1, target_modules=None):
    """Apply LoRA to specific layers in the model."""
    if target_modules is None:
        target_modules = ["backbone.body.layer3", "backbone.body.layer4"]

    lora_layers = []

    def replace_with_lora(parent, name, module):
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
            lora_layer = LoRALayer(module, rank, alpha, dropout)
            setattr(parent, name, lora_layer)
            lora_layers.append(lora_layer)
            return True
        return False

    def recursive_apply(parent, prefix=""):
        for name, module in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            should_apply = any(target in full_name for target in target_modules)

            if should_apply and isinstance(module, nn.Conv2d):
                if replace_with_lora(parent, name, module):
                    print(f"  Applied LoRA to: {full_name}")
            else:
                recursive_apply(module, full_name)

    recursive_apply(model)
    return lora_layers


def load_model(model_path, device):
    """Load the LoRA finetuned model."""
    print("Loading model architecture...")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace head
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Apply LoRA (must match training architecture)
    print("Applying LoRA layers...")
    apply_lora_to_model(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        target_modules=["backbone.body.layer3", "backbone.body.layer4"]
    )

    # Load weights
    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def evaluate_model(model, coco_gt, device):
    """Run inference and return results list."""
    valid_ids = set(coco_gt.getImgIds())
    results_list = []

    for seq_idx in tqdm(VAL_SEQS, desc="Evaluating"):
        folder = Path(DATASET_PATH) / f"{seq_idx:04d}"
        if not folder.exists():
            continue

        img_files = sorted(list(folder.glob("*.png")))

        for img_path in img_files:
            frame_idx = int(img_path.stem)
            unique_image_id = (seq_idx * 100000) + frame_idx

            if unique_image_id not in valid_ids:
                continue

            image = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(image).to(device)

            with torch.no_grad():
                outputs = model([img_tensor])[0]

            for score, label, bbox in zip(outputs["scores"], outputs["labels"], outputs["boxes"]):
                label_id = label.item()
                if label_id in RCNN_TO_COCO_ID:
                    coco_label = RCNN_TO_COCO_ID[label_id]
                    x1, y1, x2, y2 = bbox.tolist()
                    coco_bbox = [x1, y1, x2 - x1, y2 - y1]

                    results_list.append({
                        "image_id": unique_image_id,
                        "category_id": coco_label,
                        "bbox": coco_bbox,
                        "score": score.item()
                    })

    return results_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model(MODEL_PATH, device)

    # Load GT
    print("Loading ground truth...")
    coco_gt = COCO(GT_PATH)

    # Run evaluation
    print("\nRunning evaluation...")
    results_list = evaluate_model(model, coco_gt, device)

    print(f"\nTotal predictions: {len(results_list)}")

    # Run COCO evaluation
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    coco_evaluation(results_list, coco_gt, OUTPUT_DIR)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
