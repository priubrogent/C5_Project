import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from transformers import SamModel, SamProcessor, pipeline
from tqdm import tqdm
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS = [2, 6, 7, 8, 10, 13, 14, 16, 18]

COCO_TO_SEMANTIC = {1: 1, 3: 2}
SEMANTIC_TO_NAME = {0: "Background", 1: "Pedestrian", 2: "Car"}
NUM_CLASSES = 3

YOLO_COCO_CLASS_MAP = {0: 1, 2: 2}
YOLO_FINETUNED_CLASS_MAP = {0: 1, 1: 2}

SEMANTIC_COLORS = {
    0: (0.0, 0.0, 0.0),
    1: (0.0, 1.0, 0.0),
    2: (0.0, 0.0, 1.0),
}

TEXT_PROMPTS = {
    1: "person. pedestrian. human.",
    2: "car. vehicle. automobile.",
}

SAM_PRED_SIZE = 256


def focal_loss_multiclass(pred, target, valid=None, alpha=0.8, gamma=2.0):
    ce = F.cross_entropy(pred, target, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    if valid is not None:
        loss = loss * valid
        return loss.sum() / valid.sum().clamp(min=1)
    return loss.mean()


def dice_loss_multiclass(pred, target, num_classes=NUM_CLASSES, valid=None, smooth=1.0):
    pred_soft = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    if valid is not None:
        valid_expanded = valid.unsqueeze(1)
        pred_soft = pred_soft * valid_expanded
        target_onehot = target_onehot * valid_expanded

    dice = 0.0
    for c in range(num_classes):
        p = pred_soft[:, c].reshape(-1)
        t = target_onehot[:, c].reshape(-1)
        dice += 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    return dice / num_classes


def seg_loss_semantic(pred, target, loss_type, valid=None):
    if loss_type == "bce":
        loss = F.cross_entropy(pred, target, reduction="none")
        if valid is not None:
            loss = loss * valid
            return loss.sum() / valid.sum().clamp(min=1)
        return loss.mean()
    return focal_loss_multiclass(pred, target, valid) + dice_loss_multiclass(pred, target, valid=valid)


def build_coco_fixed(coco_path, out_dir):
    with open(coco_path) as f:
        data = json.load(f)
    img_size = {img["id"]: (img["height"], img["width"]) for img in data["images"]}
    for ann in data["annotations"]:
        segm = ann["segmentation"]
        if not isinstance(segm, dict):
            h, w = img_size[ann["image_id"]]
            ann["segmentation"] = {"counts": segm, "size": [h, w]}
    fixed_path = os.path.join(out_dir, "gt_fixed.json")
    with open(fixed_path, "w") as f:
        json.dump(data, f)
    return COCO(fixed_path), fixed_path


def instance_to_semantic_mask(coco, img_id, height, width):
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    ignore_mask = np.zeros((height, width), dtype=np.uint8)

    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        instance_mask = coco.annToMask(ann)
        cat_id = ann["category_id"]
        is_crowd = ann.get("iscrowd", 0)

        if is_crowd:
            ignore_mask[instance_mask > 0] = 1
        else:
            semantic_class = COCO_TO_SEMANTIC.get(cat_id, 0)
            if semantic_class > 0:
                semantic_mask[instance_mask > 0] = semantic_class

    return semantic_mask, ignore_mask


def get_aug_pipeline():
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.RandomScale(scale_limit=(-0.2, 0.2), p=0.5),
    ])


class KITTISemanticDataset(Dataset):

    def __init__(self, coco, image_root, seqs, processor, augment=False):
        self.coco = coco
        self.image_root = image_root
        self.processor = processor
        self.aug = get_aug_pipeline() if augment else None

        all_ids = coco.getImgIds()
        self.img_ids = []
        for iid in all_ids:
            seq_id = iid // 100000
            if seq_id not in seqs:
                continue
            ann_ids = coco.getAnnIds(imgIds=[iid])
            if ann_ids:
                fpath = os.path.join(image_root, coco.loadImgs([iid])[0]["file_name"])
                if os.path.exists(fpath):
                    self.img_ids.append(iid)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        h, w = img_info["height"], img_info["width"]

        img_path = os.path.join(self.image_root, img_info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        semantic_mask, ignore_mask = instance_to_semantic_mask(self.coco, img_id, h, w)

        if self.aug is not None:
            augmented = self.aug(image=image, masks=[semantic_mask, ignore_mask])
            image = augmented["image"]
            semantic_mask, ignore_mask = augmented["masks"]

        inputs = self.processor(
            images=Image.fromarray(image),
            return_tensors="pt",
        )

        rh = int(inputs["reshaped_input_sizes"][0, 0].item())
        rw = int(inputs["reshaped_input_sizes"][0, 1].item())

        def to_sam_space(mask_np):
            m_t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)
            resized = F.interpolate(m_t.unsqueeze(0), size=(rh, rw), mode="nearest").squeeze(0)
            padded = F.pad(resized, (0, 1024 - rw, 0, 1024 - rh))
            return F.interpolate(
                padded.unsqueeze(0),
                size=(SAM_PRED_SIZE, SAM_PRED_SIZE),
                mode="nearest"
            ).squeeze(0).squeeze(0)

        semantic_mask_sam = to_sam_space(semantic_mask).long()
        ignore_mask_sam = (to_sam_space(ignore_mask) > 0.5).float()

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "original_sizes": inputs["original_sizes"].squeeze(0),
            "reshaped_input_sizes": inputs["reshaped_input_sizes"].squeeze(0),
            "semantic_mask": semantic_mask_sam,
            "ignore_mask": ignore_mask_sam,
            "img_id": img_id,
        }


def collate_fn(batch):
    pixel_values_list = []
    original_sizes_list = []
    reshaped_sizes_list = []
    semantic_mask_list = []
    ignore_mask_list = []
    img_ids = []

    for item in batch:
        pixel_values_list.append(item["pixel_values"])
        original_sizes_list.append(item["original_sizes"])
        reshaped_sizes_list.append(item["reshaped_input_sizes"])
        semantic_mask_list.append(item["semantic_mask"])
        ignore_mask_list.append(item["ignore_mask"])
        img_ids.append(item["img_id"])

    return {
        "pixel_values": torch.stack(pixel_values_list),
        "original_sizes": torch.stack(original_sizes_list),
        "reshaped_input_sizes": torch.stack(reshaped_sizes_list),
        "semantic_mask": torch.stack(semantic_mask_list),
        "ignore_mask": torch.stack(ignore_mask_list),
        "img_ids": img_ids,
    }


def apply_freeze(model, strategy):
    for p in model.sam.parameters():
        p.requires_grad_(False)

    for p in model.semantic_head.parameters():
        p.requires_grad_(True)

    if strategy != "decoder_only":
        n = int(strategy.split("_")[2])
        for block in model.sam.vision_encoder.layers[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in model.sam.vision_encoder.neck.parameters():
            p.requires_grad_(True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")


class SemanticSegmentationHead(nn.Module):

    def __init__(self, embed_dim=256, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.class_queries = nn.Embedding(num_classes - 1, embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, num_classes, 1)
        self.norm1 = nn.BatchNorm2d(128)
        self.norm2 = nn.BatchNorm2d(64)

    def forward(self, image_embeddings):
        x = F.relu(self.norm1(self.conv1(image_embeddings)))
        x = F.relu(self.norm2(self.conv2(x)))
        logits = self.conv3(x)
        return logits


class SAMSemanticModel(nn.Module):

    def __init__(self, sam_model, num_classes=NUM_CLASSES):
        super().__init__()
        self.sam = sam_model
        self.semantic_head = SemanticSegmentationHead(
            embed_dim=256,
            num_classes=num_classes
        )

    def forward(self, pixel_values):
        image_embeddings = self.sam.get_image_embeddings(pixel_values)

        image_embeddings_upsampled = F.interpolate(
            image_embeddings,
            size=(SAM_PRED_SIZE, SAM_PRED_SIZE),
            mode="bilinear",
            align_corners=False
        )

        logits = self.semantic_head(image_embeddings_upsampled)
        return logits


def train_one_epoch(model, loader, optimizer, device, loss_type, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]")
    for batch in pbar:
        pv = batch["pixel_values"].to(device)
        gt = batch["semantic_mask"].to(device)
        ignore = batch["ignore_mask"].to(device)

        logits = model(pv)

        valid = 1.0 - ignore
        loss = seg_loss_semantic(logits, gt, loss_type, valid=valid)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

    return float(np.mean(losses))


def compute_semantic_metrics(pred_mask, gt_mask, num_classes=NUM_CLASSES, ignore_mask=None):
    if ignore_mask is not None:
        valid = ignore_mask == 0
    else:
        valid = np.ones_like(gt_mask, dtype=bool)

    metrics = {}
    ious = []

    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid
        gt_c = (gt_mask == c) & valid

        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = float('nan')

        metrics[f"IoU_{SEMANTIC_TO_NAME[c]}"] = iou
        if not np.isnan(iou):
            ious.append(iou)

    metrics["mIoU"] = np.mean(ious) if ious else 0.0

    correct = ((pred_mask == gt_mask) & valid).sum()
    total = valid.sum()
    metrics["pixel_acc"] = correct / total if total > 0 else 0.0

    return metrics


def visualize_semantic_mask(image, semantic_mask, title="Semantic Segmentation"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    overlay = np.array(image).copy().astype(float) / 255.0
    alpha = 0.5

    for class_id, color in SEMANTIC_COLORS.items():
        if class_id == 0:
            continue
        mask = semantic_mask == class_id
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    axes[1].imshow(overlay)
    axes[1].set_title(title)
    axes[1].axis("off")

    legend_patches = [
        mpatches.Patch(color=SEMANTIC_COLORS[1], label="Pedestrian"),
        mpatches.Patch(color=SEMANTIC_COLORS[2], label="Car"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=11)
    plt.tight_layout()

    return fig


def text_prompted_semantic_segmentation(image, detector, sam_model, sam_processor, device):
    w, h = image.size
    semantic_mask = np.zeros((h, w), dtype=np.uint8)

    for class_id, prompt in TEXT_PROMPTS.items():
        detections = detector(image, candidate_labels=[prompt], threshold=0.25)

        if not detections:
            continue

        boxes = [
            [d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]]
            for d in detections
        ]

        inputs = sam_processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = sam_model(**inputs, multimask_output=False)

        masks = sam_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs["original_sizes"],
            reshaped_input_sizes=inputs["reshaped_input_sizes"],
        )[0]

        for k in range(len(boxes)):
            mask_np = masks[k, 0].cpu().numpy() > 0
            semantic_mask[mask_np] = class_id

    return semantic_mask


def yolo_semantic_segmentation(image, yolo_model, sam_model, sam_processor, device,
                                class_map, threshold=0.25):
    w, h = image.size
    semantic_mask = np.zeros((h, w), dtype=np.uint8)

    yolo_results = yolo_model(image, conf=threshold, verbose=False)[0]

    if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
        return semantic_mask

    raw_classes = yolo_results.boxes.cls.cpu().numpy().astype(int)
    raw_boxes = yolo_results.boxes.xyxy.cpu().numpy()

    valid_dets = []
    for cls, box in zip(raw_classes, raw_boxes):
        cat_id = class_map.get(int(cls))
        if cat_id is not None:
            valid_dets.append((box.tolist(), cat_id))

    if not valid_dets:
        return semantic_mask

    boxes = [det[0] for det in valid_dets]

    inputs = sam_processor(
        images=image,
        input_boxes=[boxes],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs, multimask_output=False)

    masks = sam_processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs["original_sizes"],
        reshaped_input_sizes=inputs["reshaped_input_sizes"],
    )[0]

    for k, (box, cat_id) in enumerate(valid_dets):
        mask_np = masks[k, 0].cpu().numpy() > 0
        semantic_mask[mask_np] = cat_id

    return semantic_mask


@torch.no_grad()
def evaluate_yolo_semantic(coco, image_root, seqs, yolo_model, sam_model, sam_processor,
                           device, output_dir, class_map, threshold=0.25, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    all_ids = coco.getImgIds()
    img_ids = [iid for iid in all_ids if (iid // 100000) in seqs]

    if max_images:
        img_ids = img_ids[:max_images]

    all_metrics = defaultdict(list)

    for idx, img_id in enumerate(tqdm(img_ids, desc="YOLO+SAM eval")):
        img_info = coco.loadImgs([img_id])[0]
        h, w = img_info["height"], img_info["width"]

        img_path = os.path.join(image_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        gt_mask, ignore_mask = instance_to_semantic_mask(coco, img_id, h, w)

        pred_mask = yolo_semantic_segmentation(
            image, yolo_model, sam_model, sam_processor, device, class_map, threshold
        )

        metrics = compute_semantic_metrics(pred_mask, gt_mask, ignore_mask=ignore_mask)
        for k, v in metrics.items():
            if not np.isnan(v):
                all_metrics[k].append(v)

        if idx < 10:
            fig = visualize_semantic_mask(image, pred_mask, "YOLO+SAM Prediction")
            seq_id, frame_id = img_id // 100000, img_id % 100000
            fig.savefig(os.path.join(qual_dir, f"pred_seq{seq_id:04d}_frame{frame_id:06d}.png"),
                       dpi=100, bbox_inches="tight")
            plt.close(fig)

            fig = visualize_semantic_mask(image, gt_mask, "Ground Truth")
            fig.savefig(os.path.join(qual_dir, f"gt_seq{seq_id:04d}_frame{frame_id:06d}.png"),
                       dpi=100, bbox_inches="tight")
            plt.close(fig)

    results = {k: np.mean(v) for k, v in all_metrics.items()}

    print("\n=== YOLO+SAM Semantic Segmentation Results ===")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


@torch.no_grad()
def evaluate_text_prompted(coco, image_root, seqs, detector, sam_model, sam_processor,
                          device, output_dir, max_images=None):
    os.makedirs(output_dir, exist_ok=True)
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    all_ids = coco.getImgIds()
    img_ids = [iid for iid in all_ids if (iid // 100000) in seqs]

    if max_images:
        img_ids = img_ids[:max_images]

    all_metrics = defaultdict(list)

    for idx, img_id in enumerate(tqdm(img_ids, desc="Text-prompted eval")):
        img_info = coco.loadImgs([img_id])[0]
        h, w = img_info["height"], img_info["width"]

        img_path = os.path.join(image_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        gt_mask, ignore_mask = instance_to_semantic_mask(coco, img_id, h, w)

        pred_mask = text_prompted_semantic_segmentation(
            image, detector, sam_model, sam_processor, device
        )

        metrics = compute_semantic_metrics(pred_mask, gt_mask, ignore_mask=ignore_mask)
        for k, v in metrics.items():
            if not np.isnan(v):
                all_metrics[k].append(v)

        if idx < 10:
            fig = visualize_semantic_mask(image, pred_mask, "Text-Prompted Prediction")
            seq_id, frame_id = img_id // 100000, img_id % 100000
            fig.savefig(os.path.join(qual_dir, f"pred_seq{seq_id:04d}_frame{frame_id:06d}.png"),
                       dpi=100, bbox_inches="tight")
            plt.close(fig)

            fig = visualize_semantic_mask(image, gt_mask, "Ground Truth")
            fig.savefig(os.path.join(qual_dir, f"gt_seq{seq_id:04d}_frame{frame_id:06d}.png"),
                       dpi=100, bbox_inches="tight")
            plt.close(fig)

    results = {k: np.mean(v) for k, v in all_metrics.items()}

    print("\n=== Text-Prompted Semantic Segmentation Results ===")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def train_semantic_model(args, coco, device):
    aug_tag = "aug" if args.augment else "noaug"
    model_tag = args.model_id.split("/")[-1]
    run_name = f"{args.freeze}_{args.loss}_{aug_tag}_{model_tag}_semantic"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Device: {device}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or run_name,
            config={
                "model_id": args.model_id,
                "freeze": args.freeze,
                "loss": args.loss,
                "augment": args.augment,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_epochs": args.warmup_epochs,
                "weight_decay": args.weight_decay,
            },
        )

    print("Building datasets...")
    processor = SamProcessor.from_pretrained(args.model_id)
    train_ds = KITTISemanticDataset(coco, args.dataset_root, TRAIN_SEQS, processor, augment=args.augment)
    val_ds = KITTISemanticDataset(coco, args.dataset_root, VAL_SEQS, processor, augment=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )

    print(f"Loading {args.model_id}...")
    sam_model = SamModel.from_pretrained(args.model_id)
    model = SAMSemanticModel(sam_model, num_classes=NUM_CLASSES).to(device)
    apply_freeze(model, args.freeze)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    warmup_epochs = max(1, args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_epochs)
            ),
        ],
        milestones=[warmup_epochs],
    )

    history = {"args": vars(args), "epochs": []}
    best_miou = 0.0
    best_epoch = 0

    print("\n=== Epoch 0 / pretrained baseline ===")
    init_metrics = evaluate_finetuned_model(model, val_ds, processor, device, out_dir,
                                            "epoch00", qual_images=args.qual_images,
                                            loss_type=args.loss)
    init_miou = init_metrics.get("mIoU", 0.0)
    init_val_loss = init_metrics.get("val_loss", 0.0)
    print(f"  mIoU={init_miou:.4f}  pixel_acc={init_metrics.get('pixel_acc', 0):.4f}  val_loss={init_val_loss:.4f}")
    print(f"  IoU_Pedestrian={init_metrics.get('IoU_Pedestrian', 0):.4f}"
          f"  IoU_Car={init_metrics.get('IoU_Car', 0):.4f}")
    history["epochs"].append({"epoch": 0, "train_loss": None, "val": init_metrics})
    if not args.no_wandb:
        wandb.log({"epoch": 0, **{f"val/{k}": v for k, v in init_metrics.items()}})

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.loss, epoch)
        scheduler.step()

        print(f"\n=== Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} ===")
        val_metrics = evaluate_finetuned_model(model, val_ds, processor, device, out_dir,
                                               f"epoch{epoch+1:02d}", qual_images=args.qual_images,
                                               loss_type=args.loss)

        cur_miou = val_metrics.get("mIoU", 0.0)
        cur_val_loss = val_metrics.get("val_loss", 0.0)
        print(f"  mIoU={cur_miou:.4f}  pixel_acc={val_metrics.get('pixel_acc', 0):.4f}  val_loss={cur_val_loss:.4f}")
        print(f"  IoU_Pedestrian={val_metrics.get('IoU_Pedestrian', 0):.4f}"
              f"  IoU_Car={val_metrics.get('IoU_Car', 0):.4f}")

        history["epochs"].append({"epoch": epoch+1, "train_loss": train_loss, "val": val_metrics})

        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/lr": scheduler.get_last_lr()[0],
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })

        torch.save(model.state_dict(), os.path.join(out_dir, f"checkpoint_ep{epoch+1:02d}.pt"))

        if cur_miou > best_miou:
            best_miou, best_epoch = cur_miou, epoch + 1
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            print(f"  *** New best: mIoU={best_miou:.4f}")

    history["best_epoch"] = best_epoch
    history["best_miou"] = best_miou

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if not args.no_wandb:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_mIoU"] = best_miou
        wandb.finish()

    print(f"\nDone. Best mIoU={best_miou:.4f} at epoch {best_epoch}")
    print(f"Results saved to {out_dir}")
    return model


@torch.no_grad()
def evaluate_finetuned_model(model, dataset, processor, device, output_dir,
                             epoch_label, qual_images=5, loss_type="focal_dice"):
    model.eval()

    qual_dir = os.path.join(output_dir, "qualitative", epoch_label)
    os.makedirs(qual_dir, exist_ok=True)

    all_metrics = defaultdict(list)
    all_losses = []
    qual_count = 0

    for idx in tqdm(range(len(dataset)), desc=f"[{epoch_label}] eval"):
        item = dataset[idx]
        img_id = item["img_id"]

        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        gt_mask_tensor = item["semantic_mask"].unsqueeze(0).to(device)
        ignore_mask_tensor = item["ignore_mask"].unsqueeze(0).to(device)
        gt_mask = item["semantic_mask"].numpy()
        ignore_mask = item["ignore_mask"].numpy()

        logits = model(pixel_values)

        # Compute validation loss
        valid = 1.0 - ignore_mask_tensor
        loss = seg_loss_semantic(logits, gt_mask_tensor, loss_type, valid=valid)
        all_losses.append(loss.item())

        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

        metrics = compute_semantic_metrics(pred_mask, gt_mask, ignore_mask=ignore_mask)
        for k, v in metrics.items():
            if not np.isnan(v):
                all_metrics[k].append(v)

        if qual_count < qual_images:
            img_info = dataset.coco.loadImgs([img_id])[0]
            orig_img = Image.open(os.path.join(dataset.image_root, img_info["file_name"])).convert("RGB")

            oh, ow = img_info["height"], img_info["width"]
            pred_mask_orig = F.interpolate(
                torch.from_numpy(pred_mask).float().unsqueeze(0).unsqueeze(0),
                size=(oh, ow),
                mode="nearest"
            )[0, 0].numpy().astype(np.uint8)

            gt_mask_orig, _ = instance_to_semantic_mask(dataset.coco, img_id, oh, ow)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            axes[0].imshow(orig_img)
            axes[0].set_title("Original")
            axes[0].axis("off")

            gt_overlay = np.array(orig_img).copy().astype(float) / 255.0
            for class_id, color in SEMANTIC_COLORS.items():
                if class_id == 0:
                    continue
                mask = gt_mask_orig == class_id
                for c in range(3):
                    gt_overlay[:, :, c] = np.where(mask, gt_overlay[:, :, c] * 0.5 + color[c] * 0.5, gt_overlay[:, :, c])
            axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            pred_overlay = np.array(orig_img).copy().astype(float) / 255.0
            for class_id, color in SEMANTIC_COLORS.items():
                if class_id == 0:
                    continue
                mask = pred_mask_orig == class_id
                for c in range(3):
                    pred_overlay[:, :, c] = np.where(mask, pred_overlay[:, :, c] * 0.5 + color[c] * 0.5, pred_overlay[:, :, c])
            axes[2].imshow(pred_overlay)
            axes[2].set_title("Predicted")
            axes[2].axis("off")

            legend_patches = [
                mpatches.Patch(color=SEMANTIC_COLORS[1], label="Pedestrian"),
                mpatches.Patch(color=SEMANTIC_COLORS[2], label="Car"),
            ]
            fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=11)
            plt.tight_layout()

            seq_id, frame_id = img_id // 100000, img_id % 100000
            fig.savefig(os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png"),
                       dpi=100, bbox_inches="tight")
            plt.close(fig)
            qual_count += 1

    results = {k: round(np.mean(v), 4) for k, v in all_metrics.items()}
    if all_losses:
        results["val_loss"] = round(float(np.mean(all_losses)), 4)
    return results


def compare_approaches(args, coco, device):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    processor = SamProcessor.from_pretrained(args.model_id)

    print("\n=== Evaluating Text-Prompted Approach ===")
    detector = pipeline(
        model="IDEA-Research/grounding-dino-tiny",
        task="zero-shot-object-detection",
        device=device,
    )
    sam_model = SamModel.from_pretrained(args.model_id).to(device)
    sam_model.eval()

    text_results = evaluate_text_prompted(
        coco, args.dataset_root, VAL_SEQS, detector, sam_model, processor,
        device, os.path.join(out_dir, "text_prompted"), max_images=args.max_images
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        print("\n=== Evaluating Fine-tuned Approach ===")
        base_sam = SamModel.from_pretrained(args.model_id)
        model = SAMSemanticModel(base_sam, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
        model.eval()

        val_ds = KITTISemanticDataset(coco, args.dataset_root, VAL_SEQS, processor, augment=False)
        finetuned_results = evaluate_finetuned_model(
            model, val_ds, processor, device,
            os.path.join(out_dir, "finetuned"), "comparison", qual_images=args.qual_images
        )
    else:
        finetuned_results = None
        print("\nNo checkpoint provided for fine-tuned model comparison.")

    print("\n" + "="*60)
    print("COMPARISON: Text-Prompted vs Fine-tuned Semantic Segmentation")
    print("="*60)
    print(f"{'Metric':<20} {'Text-Prompted':>15} {'Fine-tuned':>15}")
    print("-"*60)

    for metric in ["mIoU", "pixel_acc", "IoU_Background", "IoU_Pedestrian", "IoU_Car"]:
        text_val = text_results.get(metric, float('nan'))
        ft_val = finetuned_results.get(metric, float('nan')) if finetuned_results else float('nan')
        print(f"{metric:<20} {text_val:>15.4f} {ft_val:>15.4f}")

    print("="*60)

    comparison = {
        "text_prompted": text_results,
        "finetuned": finetuned_results,
    }
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", type=str, default="train",
                   choices=["train", "text_prompt", "yolo", "eval", "compare", "compare_detectors"])

    p.add_argument("--model_id", type=str, default="facebook/sam-vit-base")
    p.add_argument("--checkpoint", type=str, default=None)

    p.add_argument("--freeze", default="decoder_only",
                   choices=["decoder_only", "decoder_and_2_vit", "decoder_and_4_vit"])
    p.add_argument("--loss", default="focal_dice", choices=["focal_dice", "bce"])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--qual_images", type=int, default=5)

    p.add_argument("--yolo_weights", type=str, default="yolo11x.pt")
    p.add_argument("--yolo_finetuned", action="store_true")
    p.add_argument("--yolo_finetuned_weights", type=str,
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/YOLO/yolo_lora_best.pt")
    p.add_argument("--yolo_threshold", type=float, default=0.25)

    p.add_argument("--max_images", type=int, default=None)

    p.add_argument("--dataset_root", type=str,
                   default="/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file", type=str,
                   default="/hhome/priubrogent/mcvpol/C5/Week1/R-CNN/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir", type=str,
                   default="/hhome/priubrogent/mcvpol/C5/Week2/outputs/task_h")

    p.add_argument("--wandb_project", default="kitti-mots-sam-semantic")
    p.add_argument("--wandb_entity", default="just-an-arbitrary-team-name")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading annotations...")
    coco, _ = build_coco_fixed(args.ann_file, args.output_dir)

    if args.mode == "train":
        train_semantic_model(args, coco, device)

    elif args.mode == "text_prompt":
        print("Loading Grounding DINO and SAM...")
        detector = pipeline(
            model="IDEA-Research/grounding-dino-tiny",
            task="zero-shot-object-detection",
            device=device,
        )
        processor = SamProcessor.from_pretrained(args.model_id)
        sam_model = SamModel.from_pretrained(args.model_id).to(device)
        sam_model.eval()

        evaluate_text_prompted(
            coco, args.dataset_root, VAL_SEQS, detector, sam_model, processor,
            device, args.output_dir, max_images=args.max_images
        )

    elif args.mode == "eval":
        if not args.checkpoint:
            print("ERROR: --checkpoint required for eval mode")
            return

        processor = SamProcessor.from_pretrained(args.model_id)
        base_sam = SamModel.from_pretrained(args.model_id)
        model = SAMSemanticModel(base_sam, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))

        val_ds = KITTISemanticDataset(coco, args.dataset_root, VAL_SEQS, processor, augment=False)
        results = evaluate_finetuned_model(model, val_ds, processor, device,
                                          args.output_dir, "eval", qual_images=args.qual_images)

        print("\n=== Fine-tuned Model Results ===")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")

    elif args.mode == "yolo":
        from ultralytics import YOLO
        try:
            from Week1.utils.utils import LoRAConv2d
        except:
            pass

        print("Loading YOLO and SAM...")
        if args.yolo_finetuned:
            yolo_model = YOLO(args.yolo_finetuned_weights)
            class_map = YOLO_FINETUNED_CLASS_MAP
            yolo_model.model.fuse = lambda *a, **kw: yolo_model.model
        else:
            yolo_model = YOLO(args.yolo_weights)
            class_map = YOLO_COCO_CLASS_MAP
        yolo_model.to(device)

        processor = SamProcessor.from_pretrained(args.model_id)
        sam_model = SamModel.from_pretrained(args.model_id).to(device)
        sam_model.eval()

        evaluate_yolo_semantic(
            coco, args.dataset_root, VAL_SEQS, yolo_model, sam_model, processor,
            device, args.output_dir, class_map, args.yolo_threshold, max_images=args.max_images
        )

    elif args.mode == "compare":
        compare_approaches(args, coco, device)

    elif args.mode == "compare_detectors":
        from ultralytics import YOLO
        try:
            from Week1.utils.utils import LoRAConv2d
        except:
            pass

        processor = SamProcessor.from_pretrained(args.model_id)
        sam_model = SamModel.from_pretrained(args.model_id).to(device)
        sam_model.eval()

        print("\n=== Evaluating Grounding DINO + SAM ===")
        detector = pipeline(
            model="IDEA-Research/grounding-dino-tiny",
            task="zero-shot-object-detection",
            device=device,
        )
        dino_results = evaluate_text_prompted(
            coco, args.dataset_root, VAL_SEQS, detector, sam_model, processor,
            device, os.path.join(args.output_dir, "grounding_dino"), max_images=args.max_images
        )

        print("\n=== Evaluating YOLO (pretrained) + SAM ===")
        yolo_model = YOLO(args.yolo_weights)
        yolo_model.to(device)
        yolo_results = evaluate_yolo_semantic(
            coco, args.dataset_root, VAL_SEQS, yolo_model, sam_model, processor,
            device, os.path.join(args.output_dir, "yolo_pretrained"),
            YOLO_COCO_CLASS_MAP, args.yolo_threshold, max_images=args.max_images
        )

        yolo_ft_results = None
        if os.path.exists(args.yolo_finetuned_weights):
            print("\n=== Evaluating YOLO (finetuned) + SAM ===")
            yolo_ft_model = YOLO(args.yolo_finetuned_weights)
            yolo_ft_model.to(device)
            yolo_ft_model.model.fuse = lambda *a, **kw: yolo_ft_model.model
            yolo_ft_results = evaluate_yolo_semantic(
                coco, args.dataset_root, VAL_SEQS, yolo_ft_model, sam_model, processor,
                device, os.path.join(args.output_dir, "yolo_finetuned"),
                YOLO_FINETUNED_CLASS_MAP, args.yolo_threshold, max_images=args.max_images
            )

        print("\n" + "="*80)
        print("COMPARISON: Grounding DINO vs YOLO for Semantic Segmentation")
        print("="*80)
        header = f"{'Metric':<20} {'Grounding DINO':>18} {'YOLO (pretrained)':>18}"
        if yolo_ft_results:
            header += f" {'YOLO (finetuned)':>18}"
        print(header)
        print("-"*80)

        for metric in ["mIoU", "pixel_acc", "IoU_Background", "IoU_Pedestrian", "IoU_Car"]:
            dino_val = dino_results.get(metric, float('nan'))
            yolo_val = yolo_results.get(metric, float('nan'))
            line = f"{metric:<20} {dino_val:>18.4f} {yolo_val:>18.4f}"
            if yolo_ft_results:
                yolo_ft_val = yolo_ft_results.get(metric, float('nan'))
                line += f" {yolo_ft_val:>18.4f}"
            print(line)

        print("="*80)

        comparison = {
            "grounding_dino": dino_results,
            "yolo_pretrained": yolo_results,
            "yolo_finetuned": yolo_ft_results,
        }
        with open(os.path.join(args.output_dir, "detector_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
