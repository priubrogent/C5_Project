from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

KITTI_TO_COCO = {1: 3, 2: 1} # Mapping from KITTI MOTS class IDs to COCO class IDs
COCO_TO_DETR_ID = {1: 0, 3: 1}
DETR_TO_COCO_ID = {0: 1, 1: 3} # Reverse mapping from DETR class IDs to COCO class IDs
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
        

def coco_evaluation(results_list, coco_gt, output_path):
    """
    Perform COCO evaluation and save results with 3 decimal precision.
    Combines global and per-class metrics into a single JSON output.

    Args:
        results_list (list): List of detection results in COCO format.
        coco_gt (COCO): COCO object containing ground truth annotations.
        output_path (str): Path to save the evaluation results.
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
    output_file = os.path.join(output_path, "evaluation_metrics.json")
    with open(output_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {output_file}")
        
        
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

    
def plot_loss(trainer, output_dir):
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

    # Initialize figure
    plt.figure(figsize=(12, 8))
    
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
    save_path = os.path.join(output_dir, "loss_curve_epochs.png")
    plt.savefig(save_path, dpi=300)
    print(f"Epoch-based loss curve saved to: {save_path}")