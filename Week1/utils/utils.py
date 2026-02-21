from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
from PIL import ImageDraw, ImageFont

KITTI_TO_COCO = {1: 3, 2: 1} # Mapping from KITTI MOTS class IDs to COCO class IDs
PERSON_ID = 1  # COCO ID for 'person'
CAR_ID = 3     # COCO ID for 'car'
COCO_CLASSES = set(KITTI_TO_COCO.values())

# Split for Validation and Training done in the original KITTI-MOTS paper.
TRAIN_SEQS = [0,1,3,4,5,9,11,12,15,17,19,20]
VAL_SEQS = [2,6,7,8,10,13,14,16,18]

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

    Args:
        results_list (list): List of detection results in COCO format.
        coco_gt (COCO): COCO object containing ground truth annotations.
        output_path (str): Path to save the evaluation results.
    """
    print("\n--- Running COCO Evaluation ---")
    
    # Load predictions directly from the list
    coco_dt = coco_gt.loadRes(results_list)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Filter evaluation to only include our target classes
    coco_eval.params.catIds = list(COCO_CLASSES) 
    
    # Limit evaluation to the images we actually processed
    processed_ids = list(set(r['image_id'] for r in results_list))
    coco_eval.params.imgIds = processed_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Save Metrics to JSON
    stats = coco_eval.stats.tolist()
    metric_names = ["mAP_0.50_0.95", "mAP_0.50", "mAP_0.75", "mAP_s", "mAP_m", "mAP_l", 
                    "mAR_1", "mAR_10", "mAR_100", "mAR_s", "mAR_m", "mAR_l"]
    
    # Apply rounding to 3 decimal places for each metric
    final_metrics = {name: round(float(stat), 3) for name, stat in zip(metric_names, stats)}

    with open(os.path.join(output_path, "evaluation_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=4)
        
def filter_results(scores, labels, boxes):
    """
    Filter results to only include valid classes and prepare for drawing.

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