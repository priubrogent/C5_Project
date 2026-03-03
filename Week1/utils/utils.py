from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

KITTI_TO_COCO = {1: 3, 2: 1} # Mapping from KITTI MOTS class IDs to COCO class IDs
COCO_TO_DETR_ID = {1: 0, 3: 1}
DETR_TO_COCO_ID = {0: 1, 1: 3} # Reverse mapping from DETR class IDs to COCO class IDs
COCO_TO_RCNN_ID = {1: 1, 3: 2}  # COCO person(1)->RCNN(1), COCO car(3)->RCNN(2), 0 is background
RCNN_TO_COCO_ID = {1: 1, 2: 3}  # Reverse mapping from R-CNN IDs to COCO IDs
PERSON_ID = 1  # COCO ID for 'person'
CAR_ID = 3     # COCO ID for 'car'
COCO_CLASSES = set(KITTI_TO_COCO.values())

# Split for Validation and Training done in the original KITTI-MOTS paper.
TRAIN_SEQS = [0,1,3,4,5,9,11,12,15,17,19,20]
VAL_SEQS = [2,6,7,8,10,13,14,16,18]

# Font size settings for the plot
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS = 20
FONT_SIZE_TICKS = 18
FONT_SIZE_LEGEND = 18

def draw_bboxes(image, bboxes, labels, scores=None, label_map=None, threshold=0.5, box_type="pred"):
    """
    Universal function to draw bounding boxes for Predictions, GT, or Ignore regions.

    Args:
        image (PIL.Image): The input image.
        bboxes (list): List of boxes. Format [x_min, y_min, x_max, y_max] or [x, y, w, h].
        labels (list): List of class IDs.
        scores (list, optional): Confidence scores. If None (for GT), 1.0 is assumed.
        label_map (dict): Mapping from class IDs to names (e.g., {1: 'person', 3: 'car'}).
        threshold (float): Score threshold to filter boxes.
        box_type (str): "pred" (Red), "gt" (Green), or "ignore" (Orange).
    """
    draw = ImageDraw.Draw(image)
    
    # 1. Define Styling based on box_type
    styles = {
        "pred":   {"color": (255, 0, 0),   "label_prefix": "P"},  # Red
        "gt":     {"color": (0, 255, 0),   "label_prefix": "GT"}, # Green
        "ignore": {"color": (255, 165, 0), "label_prefix": "IGN"} # Orange
    }
    style = styles.get(box_type, styles["pred"])
    
    # Ensure scores exist for the loop
    if scores is None:
        scores = [1.0] * len(bboxes)

    for bbox, label, score in zip(bboxes, labels, scores):
        # 2. Threshold Filtering
        if score < threshold:
            continue
        
        # 3. Handle different bbox formats
        # If the box is [x, y, w, h] (COCO/KITTI-MOTS style), convert to [x1, y1, x2, y2]
        if box_type in ["gt", "ignore"]:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1, y1, x2, y2 = bbox

        # 4. Prepare Label Text
        cat_name = label_map.get(label, f"ID:{label}")
        if box_type == "pred":
            label_text = f"{style['label_prefix']}-{cat_name}: {round(float(score), 2)}"
        elif box_type == "gt":
            label_text = f"{style['label_prefix']}-{cat_name}"
        else:
            label_text = f"{style['label_prefix']}"

        # 5. Draw Rectangle
        draw.rectangle([x1, y1, x2, y2], outline=style["color"], width=3)
        
        # 6. Draw Label Background and Text
        # Using a small offset so text doesn't sit exactly on the line
        text_pos = (x1, y1 - 2)
        try:
            text_bbox = draw.textbbox(text_pos, label_text, font_size=14, anchor="ls")
            draw.rectangle(text_bbox, fill=style["color"])
            draw.text(text_pos, label_text, fill="black" if box_type!="pred" else "white", font_size=14, anchor="ls")
        except AttributeError:
            # Fallback for older PIL versions
            draw.text(text_pos, label_text, fill=style["color"])

    return image
        

def coco_evaluation(results_list, coco_gt, output_path, file_name="evaluation_results.json", save=True):
    """
    Perform COCO evaluation and save results with 3 decimal precision.
    Combines global and per-class metrics into a single JSON output.

    Args:
        results_list (list): List of detection results in COCO format.
        coco_gt (COCO): COCO object containing ground truth annotations.
        output_path (str): Path to save the evaluation results.
    Returns:
        dict: A dictionary containing all evaluation metrics.
    """
    print("\n--- Running Detailed COCO Evaluation (Fixed Dimensions) ---")
    
    coco_dt = coco_gt.loadRes(results_list)

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    target_cat_ids = list(COCO_CLASSES) 
    coco_eval.params.catIds = target_cat_ids
    processed_ids = list(set(r['image_id'] for r in results_list))
    coco_eval.params.imgIds = processed_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 1. Global Metrics (COCO standard)
    stats = coco_eval.stats.tolist()
    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75", "mAP_s", "mAP_m", "mAP_l", 
        "mAR_1", "mAR_10", "mAR_100", "mAR_s", "mAR_m", "mAR_l"
    ]
    final_metrics = {name: round(float(stat), 3) for name, stat in zip(metric_names, stats)}

    # 2. Corrected Per-Class Metrics
    precision = coco_eval.eval['precision'] # [T, R, K, A, M]
    recall = coco_eval.eval['recall']       # [T, K, A, M]
    
    cat_id_to_name = {1: "Pedestrian", 3: "Car"}
    size_map = {0: "all", 1: "small", 2: "medium", 3: "large"}

    # We use index 2 (or -1) for the last dimension to get MaxDets=100
    for k_idx, cat_id in enumerate(target_cat_ids):
        class_name = cat_id_to_name.get(cat_id, f"Class_{cat_id}")
        
        for a_idx, size_label in size_map.items():
            # Take all the detections of the model (100)
            s_prec = precision[:, :, k_idx, a_idx, 2] 
            mAP_val = np.mean(s_prec[s_prec > -1]) if np.any(s_prec > -1) else 0.0
            final_metrics[f"mAP_0.50_0.95_{class_name}_{size_label}"] = round(float(mAP_val), 3)
            
            # Take all the detections of the model (100)
            s_rec = recall[:, k_idx, a_idx, 2]
            mAR_val = np.mean(s_rec[s_rec > -1]) if np.any(s_rec > -1) else 0.0
            final_metrics[f"mAR_100_{class_name}_{size_label}"] = round(float(mAR_val), 3)

    # Save to JSON
    if save:
        output_file = os.path.join(output_path, file_name)
        with open(output_file, "w") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Saved evaluation metrics to: {output_file}")

        
    return final_metrics
        

def filter_results(scores, labels, boxes):
    """
    Filter results to only include valid classes (car or pedestrian) and prepare for drawing.

    Args:
        scores (list): List of confidence scores.
        labels (list): List of class IDs.
        boxes (list): List of bounding boxes.
    Returns:
        valid_boxes, valid_labels, valid_scores: Filtered lists for valid classes.
    """
    valid_boxes, valid_labels, valid_scores = [], [], []
    
    for score, label, bbox in zip(scores, labels, boxes):
        label_id = label.item()
        if label_id in COCO_CLASSES:
            valid_boxes.append(bbox.tolist())
            valid_labels.append(label_id)
            valid_scores.append(score.item())
            
    return valid_boxes, valid_labels, valid_scores

    
def plot_loss(trainer, output_dir, file_name="loss_curve.png", save=True):
    """
    Plot training and validation loss curves using Epochs as the x-axis.

    Args:
        trainer (Trainer): The Hugging Face Trainer object after training.
        output_dir (str): Directory where the plot will be saved.
    """
    # Extract logs from the trainer state
    history = trainer.state.log_history
    
    # Extract training data using the 'epoch' key instead of 'step'
    train_loss = [log["loss"] for log in history if "loss" in log]
    train_epochs = [log["epoch"] for log in history if "loss" in log]
    
    # Extract validation data
    val_loss = [log["eval_loss"] for log in history if "eval_loss" in log]
    val_epochs = [log["epoch"] for log in history if "eval_loss" in log]
    
    fig = plt.figure(figsize=(12, 8))
    
    # Plot curves using epochs on the X-axis
    plt.plot(train_epochs, train_loss, label="Training Loss", color="blue", lw=3)
    
    if val_loss:
        # Markers help distinguish the validation points usually taken at integer epochs
        plt.plot(val_epochs, val_loss, label="Validation Loss", color="red", lw=3, marker='o', markersize=8)

    # Apply specific font sizes and labels
    plt.xlabel("Epochs", fontsize=FONT_SIZE_AXIS, labelpad=15)
    plt.ylabel("Loss", fontsize=FONT_SIZE_AXIS, labelpad=15)
    plt.title("DETR LoRA Training Progress", fontsize=FONT_SIZE_TITLE, pad=20)
    
    # Styling
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save high-resolution output
    if save:
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300)
        print(f"Epoch-based loss curve saved to: {save_path}") 
           
    return fig

def visualize_gt_vs_preds(coco_gt, results_list, images_path, output_dir, class_ids=[1, 3], show=False):
    """
    Visualize GT and YOLO predictions for the given images.
    
    Args:
        coco_gt (COCO): COCO ground truth object.
        results_list (list): List of prediction dicts in COCO format.
        images_path (str or Path): Root folder where images are stored.
        output_dir (str): Folder to save visualization images.
        class_ids (list): List of COCO class IDs to visualize (default: [1, 3]).
        show (bool): If True, displays images using PIL.Image.show()
    """
    #os.makedirs(output_dir, exist_ok=True)
    
    # Organize predictions by image_id for fast lookup
    preds_by_img = {}
    for r in results_list:
        if r['category_id'] not in class_ids:
            continue
        preds_by_img.setdefault(r['image_id'], []).append(r)
    
    # Iterate over all images with GT annotations
    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        img_info = coco_gt.loadImgs([img_id])[0]
        img_filename = img_info['file_name']
        img_path = os.path.join(images_path, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        image = Image.open(img_path).convert("RGB")
        
        # Load GT annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        
        gt_boxes, gt_labels = [], []
        for ann in anns:
            if ann['category_id'] in class_ids:
                gt_boxes.append(ann['bbox'])
                gt_labels.append(ann['category_id'])
        
        # Load predictions
        pred_boxes, pred_labels, pred_scores = [], [], []
        for r in preds_by_img.get(img_id, []):
            pred_boxes.append(r['bbox'])
            pred_labels.append(r['category_id'])
            pred_scores.append(r['score'])
        
        # Draw GT (Green) and Predictions (Red)
        if gt_boxes:
            image = draw_bboxes(image, gt_boxes, gt_labels, scores=None, label_map={1: "Pedestrian", 3: "Car"}, box_type="gt")
        if pred_boxes:
            image = draw_bboxes(image, pred_boxes, pred_labels, scores=pred_scores, label_map={1: "Pedestrian", 3: "Car"}, box_type="pred")
        
        '''save_path = os.path.join(output_dir, f"{img_filename}")
        image.save(save_path)'''
        # Show image and block until closed
         # This blocks execution until window is closed
        
        #print(f"Saved visualization for image {img_filename} at {save_path}")


def compute_iou(boxA, boxB):
    """
    Compute IoU between two boxes in COCO format [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0

    return interArea / unionArea


def debug_single_image_detection(coco_gt, results_list, image_id, iou_threshold=0.5):
    """
    Debug detection results for a single image.
    Prints TP, FP, FN, Precision, Recall, F1, mean IoU.
    """

    print(f"\n========== DEBUG IMAGE {image_id} ==========")

    # --- Load GT ---
    ann_ids = coco_gt.getAnnIds(imgIds=[image_id])
    anns = coco_gt.loadAnns(ann_ids)

    gt_boxes = []
    gt_labels = []

    for ann in anns:
        if ann['category_id'] in COCO_CLASSES:
            gt_boxes.append(ann['bbox'])
            gt_labels.append(ann['category_id'])

    # --- Load Predictions ---
    preds = [r for r in results_list if r["image_id"] == image_id]

    pred_boxes = [r["bbox"] for r in preds]
    pred_labels = [r["category_id"] for r in preds]
    pred_scores = [r["score"] for r in preds]

    matched_gt = set()
    TP = 0
    FP = 0
    ious = []

    # --- Matching ---
    for p_box, p_label in zip(pred_boxes, pred_labels):

        best_iou = 0
        best_gt_idx = -1

        for i, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if i in matched_gt:
                continue
            if p_label != g_label:
                continue

            iou = compute_iou(p_box, g_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
        else:
            FP += 1

    FN = len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(ious) if ious else 0

    print(f"GT objects: {len(gt_boxes)}")
    print(f"Pred objects: {len(pred_boxes)}")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"Mean IoU (matched): {mean_iou:.3f}")
    print("=====================================\n")

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "Mean_IoU": round(mean_iou, 3)
    }


class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.alpha = alpha

        out_channels = conv_layer.out_channels
        in_channels = conv_layer.in_channels
        kH, kW = conv_layer.kernel_size
        groups = conv_layer.groups

        self.groups = groups
        self.kH = kH
        self.kW = kW

        in_per_group = in_channels // groups

        # Freeze original conv weights
        for param in self.conv.parameters():
            param.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_per_group * kH * kW))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank))
        self.scaling = alpha / rank

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = self.conv(x)
        weight_delta = torch.matmul(self.lora_B, self.lora_A)
        weight_delta = weight_delta.view(
            self.conv.out_channels,
            self.conv.in_channels // self.groups,
            self.kH,
            self.kW
        ) * self.scaling

        lora_out = F.conv2d(
            x,
            weight_delta.to(x.device),
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.groups,
        )

        return original_out + lora_out


def inject_lora(model, rank=4, alpha=1.0):
    """
    Recursively replace Conv2d layers with LoRAConv2d in a YOLO model.
    """
    for name, module in list(model.model.named_modules()):
        if isinstance(module, nn.Conv2d):
            parent = model.model
            components = name.split(".")
            for comp in components[:-1]:
                parent = getattr(parent, comp)

            setattr(parent, components[-1], LoRAConv2d(module, rank, alpha))

    print("LoRA injected into Conv2d layers.")