"""
Visualize the first frame of each validation sequence using the finetuned
RT-DETR model from finetune_fixed3 (LoRA + smart weight initialization).

Draws:
  Red   = Predictions (P-person / P-car)
  Green = Ground truth
  Orange= Ignore regions
"""
import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, RTDetrConfig
from peft import PeftModel
from pycocotools.coco import COCO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import draw_bboxes, coco_evaluation, VAL_SEQS, DETR_TO_COCO_ID

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = "/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02"
GT_PATH      = os.path.join(os.path.dirname(__file__), "..", "kitti_mots_to_coco_gt.json")
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune_fixed3", "final_lora_adapter")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune_fixed3", "visualizations")
CHECKPOINT   = "PekingU/rtdetr_r101vd"

# The finetuned model uses 0-indexed classes (0=person, 1=car)
ID2LABEL = {0: "person", 1: "car"}
LABEL2ID = {"person": 0, "car": 1}

# GT annotations use COCO category IDs (1=person, 3=car)
GT_LABEL_MAP = {1: "person", 3: "car"}

VIS_THRESHOLD = 0.5


def load_model(device):
    # Build the 2-class base model (same config as during training)
    model_config = RTDetrConfig.from_pretrained(CHECKPOINT)
    model_config.num_labels = len(ID2LABEL)
    model_config.id2label   = ID2LABEL
    model_config.label2id   = LABEL2ID

    base_model = RTDetrForObjectDetection.from_pretrained(
        CHECKPOINT,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # Restore detection head weights — PEFT only saves LoRA matrices, so heads
    # were saved separately as detection_heads.pt during training.
    heads_path = os.path.join(ADAPTER_PATH, "detection_heads.pt")
    if os.path.exists(heads_path):
        head_state = torch.load(heads_path, map_location="cpu")
        params = dict(model.named_parameters())
        for name, data in head_state.items():
            if name in params:
                params[name].data.copy_(data)
        print(f"Loaded detection head weights from {heads_path}")
    else:
        print(f"WARNING: {heads_path} not found — detection heads have untrained weights.")

    model.to(device)
    model.eval()
    return model


def visualize_first_frames():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)
    model     = load_model(device)

    if not os.path.exists(GT_PATH):
        print(f"GT file not found: {GT_PATH}\nRun utils/GT_conversor.py first.")
        return
    coco_gt = COCO(GT_PATH)

    print(f"Generating visualizations for the first frame of {len(VAL_SEQS)} val sequences...")

    inference_times_ms = []

    for seq_idx in tqdm(VAL_SEQS, desc="Sequences"):
        img_path = Path(DATASET_PATH) / f"{seq_idx:04d}" / "000000.png"
        if not img_path.exists():
            print(f"  Skipping seq {seq_idx:04d} — image not found")
            continue

        image         = Image.open(img_path).convert("RGB")
        unique_img_id = seq_idx * 100000  # frame 0

        # --- Inference ---
        inputs = processor(images=image, return_tensors="pt").to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.time()

        with torch.no_grad():
            outputs = model(**inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        inference_times_ms.append((t_end - t_start) * 1000)

        target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
        preds = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=VIS_THRESHOLD
        )[0]

        # --- Ground truth ---
        ann_ids = coco_gt.getAnnIds(imgIds=[unique_img_id])
        anns    = coco_gt.loadAnns(ann_ids)

        gt_boxes,  gt_labels  = [], []
        ign_boxes, ign_labels = [], []
        for ann in anns:
            if ann.get("iscrowd") == 1: # or ann.get("category_id") == 10:
                ign_boxes.append(ann["bbox"])
                ign_labels.append(ann["category_id"])
            else:
                gt_boxes.append(ann["bbox"])
                gt_labels.append(ann["category_id"])

        # --- Draw ---
        # Predictions: model outputs label IDs 0/1, boxes in xyxy format
        pred_boxes  = [b.tolist() for b in preds["boxes"]]
        pred_labels = [l.item() for l in preds["labels"]]
        pred_scores = [s.item() for s in preds["scores"]]

        if pred_boxes:
            image = draw_bboxes(image, pred_boxes, pred_labels, pred_scores,
                                label_map=ID2LABEL, threshold=VIS_THRESHOLD, box_type="pred")
        if gt_boxes:
            image = draw_bboxes(image, gt_boxes, gt_labels,
                                label_map=GT_LABEL_MAP, box_type="gt")
        if ign_boxes:
            image = draw_bboxes(image, ign_boxes, ign_labels,
                                label_map=GT_LABEL_MAP, box_type="ignore")

        save_path = os.path.join(OUTPUT_DIR, f"seq_{seq_idx:04d}_finetuned.png")
        image.save(save_path)

    print(f"\nVisualizations saved to: {OUTPUT_DIR}")

    if inference_times_ms:
        print(f"\n--- Inference time (model forward pass only) ---")
        print(f"  Images timed : {len(inference_times_ms)}")
        print(f"  Mean         : {sum(inference_times_ms)/len(inference_times_ms):.1f} ms")
        print(f"  Min          : {min(inference_times_ms):.1f} ms")
        print(f"  Max          : {max(inference_times_ms):.1f} ms")


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_output_dir = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune_fixed3")
    os.makedirs(eval_output_dir, exist_ok=True)

    processor = RTDetrImageProcessor.from_pretrained(CHECKPOINT)
    model     = load_model(device)

    if not os.path.exists(GT_PATH):
        print(f"GT file not found: {GT_PATH}\nRun utils/GT_conversor.py first.")
        return
    coco_gt   = COCO(GT_PATH)
    valid_ids = set(coco_gt.getImgIds())

    results_list = []

    print(f"Running inference on {len(VAL_SEQS)} validation sequences...")
    for seq_idx in tqdm(VAL_SEQS, desc="Sequences", position=0):
        folder = Path(DATASET_PATH) / f"{seq_idx:04d}"
        if not folder.exists():
            continue

        img_files = sorted(folder.glob("*.png"))
        for img_path in tqdm(img_files, desc=f"Seq {seq_idx:04d}", position=1, leave=False):
            frame_idx     = int(img_path.stem)
            unique_img_id = seq_idx * 100000 + frame_idx

            if unique_img_id not in valid_ids:
                continue

            image  = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[image.size[1], image.size[0]]]).to(device)
            # threshold=0 required for a correct mAP PR curve
            preds = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.0
            )[0]

            for score, label, bbox in zip(preds["scores"], preds["labels"], preds["boxes"]):
                label_id = label.item()
                if label_id in DETR_TO_COCO_ID:
                    x1, y1, x2, y2 = bbox.tolist()
                    results_list.append({
                        "image_id":   unique_img_id,
                        "category_id": DETR_TO_COCO_ID[label_id],
                        "bbox":       [x1, y1, x2 - x1, y2 - y1],
                        "score":      score.item(),
                    })

    if results_list:
        print(f"\nTotal detections collected: {len(results_list)}")
        coco_evaluation(results_list, coco_gt, eval_output_dir, save=True)
    else:
        print("Warning: no detections produced on the validation set.")


if __name__ == "__main__":
    run_evaluation()
    visualize_first_frames()
