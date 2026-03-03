import os

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
os.environ["WANDB_MODE"] = "online"

import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import wandb
import torch
import time

from Week1.utils.utils import coco_evaluation, TRAIN_SEQS, VAL_SEQS, COCO_CLASSES, visualize_gt_vs_preds, filter_results, draw_bboxes, debug_single_image_detection, LoRAConv2d, inject_lora
from pycocotools.coco import COCO

KITTI_TO_COCO = {0: 1, 2: 3} # Mapping from KITTI MOTS class IDs to COCO class IDs
#Opcions possibles depenent del size: yolov10n.pt, yolov10s.pt, yolov10m.pt, yolov10l.pt, yolov10x.pt
model = YOLO("yolo26m.pt")
model.to("cuda")

DATASET_PATH = "/data/113-2/users/gasbert/master/C5/KITTI-MOTS"
IMAGES_PATH = os.path.join(DATASET_PATH, "training/image_02")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "instances_txt")

allowed_classes = set(KITTI_TO_COCO.keys())

GT_PATH = "/data/113-2/users/gasbert/master/C5/KITTI-MOTS/kitti_mots_to_coco_gt.json"
OUTPUT_DIR = "./YOLO/Results_YOLO/task_d/"


def visualize_first_frames_yolo():
    """
    Inferences the first image of each sequence and overlays YOLO predictions, GT boxes, and ignore regions.
    Execution pauses until the image window is closed.
    """
    vis_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Load GT
    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found.")
        return
    coco_gt = COCO(GT_PATH)

    print(f"Generating visualizations for the first frame of each validation sequence...")

    for seq_idx in VAL_SEQS:
        img_path = Path(IMAGES_PATH) / f"{seq_idx:04d}" / "000000.png"
        if not img_path.exists():
            continue

        # Load image
        image_cv2 = cv2.imread(str(img_path))
        if image_cv2 is None:
            continue
        image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        unique_image_id = seq_idx * 100000  # frame_idx = 0

        # --- YOLO inference ---
        results = model(image_cv2, verbose=False)
        result = results[0]

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # Keep only allowed classes
            keep_indices = [i for i, c in enumerate(classes) if c in allowed_classes]
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]

            # Convert YOLO boxes to COCO style [x, y, w, h] and map KITTI -> COCO
            pred_boxes, pred_labels, pred_scores = [], [], []
            for box, score, cls in zip(boxes, scores, classes):
                mapped_cls = KITTI_TO_COCO[cls]
                x1, y1, x2, y2 = box.tolist()
                #pred_boxes.append([x1, y1, x2 - x1, y2 - y1])
                pred_boxes.append(box.tolist())
                pred_labels.append(mapped_cls)
                pred_scores.append(score)
        else:
            pred_boxes, pred_labels, pred_scores = [], [], []

        # --- GT boxes and ignore regions ---
        ann_ids = coco_gt.getAnnIds(imgIds=[unique_image_id])
        anns = coco_gt.loadAnns(ann_ids)

        gt_boxes, gt_labels = [], []
        ign_boxes, ign_labels = [], []

        for ann in anns:
            if ann.get('iscrowd') == 1 or ann.get('category_id') == 10:
                ign_boxes.append(ann['bbox'])
                ign_labels.append(ann['category_id'])
            elif ann['category_id'] in [1, 3]:
                gt_boxes.append(ann['bbox'])
                gt_labels.append(ann['category_id'])

        # --- Draw boxes ---
        '''if gt_boxes:
            image_pil = draw_bboxes(image_pil, gt_boxes, gt_labels,
                                     box_type="gt", label_map={1: "Pedestrian", 3: "Car"})'''
        if pred_boxes:
            image_pil = draw_bboxes(image_pil, pred_boxes, pred_labels, pred_scores,
                                     box_type="pred", label_map={1: "Pedestrian", 3: "Car"})
        '''if ign_boxes:
            image_pil = draw_bboxes(image_pil, ign_boxes, ign_labels,
                                     box_type="ignore", label_map={1: "Pedestrian", 3: "Car"})'''

        # Save image
        save_path = os.path.join(vis_dir, f"seq_{seq_idx:04d}_vis.png")
        image_pil.save(save_path)
        print(f"Saved: {save_path}")

        '''# --- Show image and block until window closed ---
        plt.figure(figsize=(12, 6))
        plt.imshow(image_pil)
        plt.axis('off')
        plt.title(f"YOLO Predictions vs GT: Sequence {seq_idx:04d}")
        plt.show()  # Blocks execution until window is closed'''


def run_inference_on_dataset():
    sequence_folders = sorted(os.listdir(IMAGES_PATH))
    print(f"Found {len(sequence_folders)} sequences in the dataset.")

    for seq_folder in sequence_folders:
        seq_path = os.path.join(IMAGES_PATH, seq_folder)
        image_files = sorted(os.listdir(seq_path))
        print(f"Found {len(image_files)} images in sequence {seq_folder}.")

        for img_name in image_files:
            if not img_name.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(seq_path, img_name)
            image = cv2.imread(img_path)

            #YOLO inference
            start_time = time.perf_counter()
            results = model(image)
            end_time = time.perf_counter()

            inference_time_ms = (end_time - start_time) * 1000

            result = results[0]
            
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            #Filtrem classes no desitjades
            keep_indices = [i for i, c in enumerate(classes) if c in allowed_classes]
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]
            
            print(f"\nImage: {img_name}")
            print(f"Inference time: {inference_time_ms:.2f} ms")
            for box, score, cls in zip(boxes, scores, classes):
                print(f"Pred: Class={cls}, Conf={score:.3f}, Box={box}")

            #Visualitzem sol les prediccions que contenen les classes desitjades
            if len(keep_indices) > 0:
                filtered_result = result
                filtered_result.boxes = result.boxes[keep_indices]
                annotated_img = filtered_result.plot()
                cv2.imshow("YOLOv10 Filtered Predictions", annotated_img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


def run_evaluation():
    model.to("cuda")
    model.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(GT_PATH):
        print(f"Error: {GT_PATH} not found. Run GT_conversor.py first.")
        return

    coco_gt = COCO(GT_PATH)

    valid_ids = set(coco_gt.getImgIds())


    results_list = []

    print(f"Starting YOLO evaluation on {len(VAL_SEQS)} validation sequences...")

    for seq_idx in tqdm(VAL_SEQS, desc="Sequences", position=0):
        folder = Path(IMAGES_PATH) / f"{seq_idx:04d}"
        if not folder.exists():
            continue

        img_files = sorted(folder.glob("*.png"))

        for img_path in tqdm(img_files, desc=f"Seq {seq_idx:04d}", position=1, leave=False):
            frame_idx = int(img_path.stem)

            # Same unique ID logic as DETR
            unique_image_id = (seq_idx * 100000) + frame_idx

            if unique_image_id not in valid_ids:
                continue
            
    
            image = cv2.imread(str(img_path))

            
            # --- YOLO inference ---
            results = model(image, verbose=False)
            result = results[0]

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)            

            # --- Filter only allowed classes ---
            for box, score, cls in zip(boxes, scores, classes):
                if cls not in KITTI_TO_COCO:
                    continue

                mapped_cls = KITTI_TO_COCO[cls]

                x1, y1, x2, y2 = box.tolist()
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO format

                ann_ids = coco_gt.getAnnIds(imgIds=[unique_image_id])
                anns = coco_gt.loadAnns(ann_ids)


                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    print("Invalid box:", x1, y1, x2, y2)

                results_list.append({
                    "image_id": unique_image_id,
                    "category_id": mapped_cls,
                    "bbox": coco_bbox,
                    "score": float(score)
                })


    print("Pred categories:", set(r["category_id"] for r in results_list))
    print("GT categories:", coco_gt.getCatIds())
    # --- Run COCO Evaluation ---
    coco_evaluation(results_list, coco_gt, OUTPUT_DIR)
    '''visualize_gt_vs_preds(
        coco_gt=coco_gt,
        results_list=results_list,
        images_path=IMAGES_PATH,
        output_dir="./YOLO/Results_YOLO/visualizations",
        class_ids=[1, 3],
        show=True
    )'''


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_all(model):
    for param in model.model.parameters():
        param.requires_grad = False


def unfreeze_last_n_layers(model, n):
    """
    Unfreeze last n modules of model.model.model
    """
    total_layers = len(model.model.model)

    for i in range(total_layers - n, total_layers):
        for param in model.model.model[i].parameters():
            param.requires_grad = True


def finetune_yolo_defreeze():

    print("\n===== Starting Progressive Defreeze Fine-Tuning =====\n")

    # -------------------------
    # Initialize WandB
    # -------------------------
    wandb.init(
        project="C5_Week1_YOLO",
        name="yolo11x_progressive_defreeze",
        config={
            "model": "yolo11x.pt",
            "dataset": "KITTI-MOTS",
            "strategy": "progressive_defreeze"
        }
    )
    config = wandb.config

    # Reload fresh pretrained model
    last_checkpoint = "yolo11x.pt"  # start from pretrained
    model = YOLO("yolo11x.pt")
    model.to("cuda")

    # -------------------------
    # Stage Definitions
    # -------------------------
    stages = [
        {"name": "head_only", "unfreeze_layers": 1, "epochs": 5},
        {"name": "last_3_layers", "unfreeze_layers": 3, "epochs": 5},
        {"name": "last_6_layers", "unfreeze_layers": 6, "epochs": 5},
        {"name": "last_9_layers", "unfreeze_layers": 9, "epochs": 5},
        {"name": "last_12_layers", "unfreeze_layers": 12, "epochs": 5},
    ]

    for stage_idx, stage in enumerate(stages):

        print(f"\n--- Stage {stage_idx+1}: {stage['name']} ---")

        model = YOLO(last_checkpoint)
        model.to("cuda")

        # Freeze everything
        freeze_all(model)

        # Unfreeze desired layers
        unfreeze_last_n_layers(model, stage["unfreeze_layers"])

        trainable_params = count_trainable_params(model.model)

        print(f"Trainable parameters: {trainable_params}")

        wandb.log({
            "stage": stage_idx,
            "stage_name": stage["name"],
            "trainable_params": trainable_params
        })

        # Path where Ultralytics saves best model
        stage_save_dir = os.path.join(
            "runs/detect/YOLO_defreeze",
            stage["name"],
            "weights",
            "best.pt"
        )

        last_checkpoint = stage_save_dir
        print(f"Next stage will load: {last_checkpoint}")

        # -------------------------
        # Train
        # -------------------------
        results = model.train(
            data="/home-local/gasbert/master/C5/C5_Project/Week1/YOLO/yolo_finetuning.yaml",
            epochs=stage["epochs"],
            imgsz=640,
            batch=16,
            lr0=config.lr0,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            device=0,
            project="YOLO_defreeze",
            name=stage["name"],
            exist_ok=True,
            verbose=True
        )

        # Log training losses
        if hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                if "loss" in k:
                    wandb.log({f"train/{k}": v, "stage": stage_idx})

        # -------------------------
        # Validation
        # -------------------------
        metrics = model.val()

        box_metrics = metrics.box

        log_dict = {
            "stage": stage_idx,

            # ---- mAP metrics ----
            "val/mAP50": box_metrics.map50,
            "val/mAP50-95": box_metrics.map,

            # ---- Recall metrics ----
            "val/mAR@100": box_metrics.mr,

        }

        wandb.log(log_dict)

    wandb.finish()

    print("\n===== Progressive Defreeze Finished =====\n")

def on_train_start_freeze_base(trainer):
    print("\n[Callback] Finalizing LoRA Trainability (Re-freezing base weights)...")
    for n, p in trainer.model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    
    # Optional: Log the count to be 100% sure
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[Callback] Trainable parameters: {trainable}\n")


def finetune_yolo_lora():
    print("\n===== Starting LoRA Fine-Tuning =====\n")

    wandb.init(
        project="C5_Week1_YOLO",
        name="yolo_lora_finetuning",
        config={
            "model": "yolo11x.pt",
            "dataset": "KITTI-MOTS",
            "strategy": "LoRA",
        }
    )

    config = wandb.config
    rank = config.lora_rank
    alpha = config.lora_alpha

    model = YOLO("yolo11x.pt")

    trainer = model._smart_load("trainer")(
        overrides=dict(
            model="yolo11x.pt",
            data="/home-local/gasbert/master/C5/C5_Project/Week1/YOLO/yolo_finetuning.yaml",
            epochs=20,
            imgsz=640,
            batch=16,
            lr0=config.lr0,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            device=0,
            project="YOLO_LoRA",
            name="lora_training",
            exist_ok=True,
            freeze=None,
        )
    )

    trainer.model = trainer.get_model(weights=model.model, cfg=model.model.yaml)

    # 🔥 Inject LoRA (Conv2d layers replaced with LoRAConv2d)
    inject_lora(trainer.model, rank=rank, alpha=alpha)

    # Freeze only original conv and all non-LoRA parameters
    for n, p in trainer.model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

    # Verify trainable params
    print("\nTrainable parameters after LoRA injection:")
    for n, p in trainer.model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)

    trainable_params = count_trainable_params(trainer.model.model)
    print(f"Trainable parameters (LoRA only): {trainable_params}")
    wandb.log({"trainable_params": trainable_params})

    trainer.add_callback("on_train_start", on_train_start_freeze_base)

    # Start training
    trainer.train()

    # Validation
    metrics = trainer.validate()
    box_metrics = metrics.box

    wandb.log({
        "val/mAP50": box_metrics.map50,
        "val/mAP50-95": box_metrics.map,
        "val/mAR@100": box_metrics.mr
    })

    wandb.finish()
    print("\n===== LoRA Fine-Tuning Finished =====\n")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run YOLOv10 on KITTI-MOTS dataset")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["c", "d", "e_defreeze", "e_lora", "v"],
        help="Tasks of Week 1: c -> (YOLOv10 inference on KITTI-MOTS), \
                               d -> (YOLOv10 inference on KITTI-MOTS with quantitative COCO metrics), \
                               e_defreeze -> (YOLOv10 finetuning on KITTI-MOTS with gradually defreezing last layers), \
                               e_lora -> (YOLOv10 finetuning on KITTI-MOTS with LoRA), \
                               v -> (YOLOv10 inference on KITTI-MOTS with visualizations)"
                               
    )
    args = parser.parse_args()

    if args.task == "c":
        run_inference_on_dataset()
    elif args.task == "d":
        run_evaluation()
    elif args.task == "e_defreeze":
        finetune_yolo_defreeze()
    elif args.task == "e_lora":
        finetune_yolo_lora()
    elif args.task == "v":
        visualize_first_frames_yolo()