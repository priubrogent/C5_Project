import os, json, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import wandb

TRAIN_SEQS    = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS      = [2, 6, 7, 8, 10, 13, 14, 16, 18]
VALID_CAT_IDS = {1, 3}
CAT_NAMES     = {1: "Pedestrian", 3: "Car"}
MASK_COLORS   = {1: (0, 255, 0), 3: (0, 0, 255)}
SAM_PRED_SIZE = 256


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def focal_loss(pred, target, valid=None, alpha=0.8, gamma=2.0):
    bce     = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t     = target * torch.sigmoid(pred) + (1 - target) * (1 - torch.sigmoid(pred))
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    loss    = alpha_t * (1 - p_t) ** gamma * bce
    if valid is not None:
        loss = loss * valid
        return loss.sum() / valid.sum().clamp(min=1)
    return loss.mean()

def dice_loss(pred, target, valid=None, smooth=1.0):
    p = torch.sigmoid(pred)
    if valid is not None:
        p = p * valid
        t = target * valid
    else:
        t = target
    p = p.reshape(-1)
    t = t.reshape(-1)
    return 1 - (2 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)

def seg_loss(pred, target, loss_type, valid=None):
    if loss_type == "bce":
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        if valid is not None:
            loss = loss * valid
            return loss.sum() / valid.sum().clamp(min=1)
        return loss.mean()
    return focal_loss(pred, target, valid) + dice_loss(pred, target, valid)


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

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

def mask_to_rle(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def overlay_masks(image, masks, cat_ids, alpha=0.45):
    img = np.array(image).copy()
    for mask, cat_id in zip(masks, cat_ids):
        color = MASK_COLORS.get(cat_id, (255, 0, 0))
        for c, v in enumerate(color):
            img[:, :, c] = np.where(mask, img[:, :, c] * (1-alpha) + v * alpha, img[:, :, c])
    return Image.fromarray(img.astype(np.uint8))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def get_aug_pipeline():
    import albumentations as A
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.RandomScale(scale_limit=(-0.2, 0.2), p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["cat_ids"],
            min_area=16,
            min_visibility=0.2,
        ),
        additional_targets={"masks": "masks"},
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KITTISAMDataset(Dataset):

    def __init__(self, coco, image_root, seqs, processor, augment=False):
        self.coco       = coco
        self.image_root = image_root
        self.processor  = processor
        self.aug        = get_aug_pipeline() if augment else None

        all_ids = coco.getImgIds()
        self.img_ids = []
        for iid in all_ids:
            if (iid // 100000) not in seqs:
                continue
            if not coco.getAnnIds(imgIds=[iid], catIds=list(VALID_CAT_IDS)):
                continue
            fpath = os.path.join(image_root, coco.loadImgs([iid])[0]["file_name"])
            if os.path.exists(fpath):
                self.img_ids.append(iid)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        image    = np.array(
            Image.open(os.path.join(self.image_root, img_info["file_name"])).convert("RGB")
        )

        ann_ids  = self.coco.getAnnIds(imgIds=[img_id], catIds=list(VALID_CAT_IDS))
        all_anns = self.coco.loadAnns(ann_ids)

        anns        = [a for a in all_anns
                       if a.get("iscrowd", 0) == 0 and a["bbox"][2] >= 2 and a["bbox"][3] >= 2]
        ignore_anns = [a for a in all_anns if a.get("iscrowd", 0) == 1]

        if not anns:
            return self.__getitem__((idx + 1) % len(self))

        boxes   = [[a["bbox"][0], a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                   for a in anns]
        masks   = [self.coco.annToMask(a) for a in anns]
        cat_ids = [a["category_id"] for a in anns]

        ignore_np = np.zeros_like(masks[0])
        for ia in ignore_anns:
            ignore_np = np.maximum(ignore_np, self.coco.annToMask(ia))

        if self.aug is not None:
            r = self.aug(image=image, bboxes=boxes, cat_ids=cat_ids,
                         masks=masks + [ignore_np])
            if not r["bboxes"]:
                return self.__getitem__((idx + 1) % len(self))
            aug_masks   = r["masks"]
            ignore_np   = aug_masks[-1]
            masks       = aug_masks[:-1]
            image, boxes, cat_ids = r["image"], list(r["bboxes"]), r["cat_ids"]

        inputs = self.processor(
            images=Image.fromarray(image),
            input_boxes=[boxes],
            return_tensors="pt",
        )

        # The processor may cap the number of boxes; align masks/cat_ids accordingly
        n_boxes = inputs["input_boxes"].shape[1]
        masks    = masks[:n_boxes]
        cat_ids  = cat_ids[:n_boxes]

        rh = int(inputs["reshaped_input_sizes"][0, 0].item())
        rw = int(inputs["reshaped_input_sizes"][0, 1].item())

        def to_sam_space(m_t):
            resized = F.interpolate(m_t.unsqueeze(0), size=(rh, rw), mode="nearest").squeeze(0)
            padded  = F.pad(resized, (0, 1024 - rw, 0, 1024 - rh))
            return F.interpolate(padded.unsqueeze(0),
                                 size=(SAM_PRED_SIZE, SAM_PRED_SIZE),
                                 mode="nearest").squeeze(0)

        masks_t  = torch.from_numpy(np.stack(masks).astype(np.float32))
        gt_masks = to_sam_space(masks_t)

        ignore_t    = torch.from_numpy(ignore_np.astype(np.float32)).unsqueeze(0)
        ignore_mask = (to_sam_space(ignore_t) > 0.5).float()

        return {
            "pixel_values":         inputs["pixel_values"].squeeze(0),
            "input_boxes":          inputs["input_boxes"].squeeze(0).reshape(-1, 1, 4),
            "original_sizes":       inputs["original_sizes"].squeeze(0),
            "reshaped_input_sizes": inputs["reshaped_input_sizes"].squeeze(0),
            "gt_masks":             gt_masks,
            "ignore_mask":          ignore_mask,
            "cat_ids":              torch.tensor(cat_ids, dtype=torch.long),
            "img_id":               img_id,
        }


def collate_fn(batch):
    max_n = max(item["gt_masks"].shape[0] for item in batch)

    pixel_values_list, input_boxes_list, gt_masks_list = [], [], []
    ignore_mask_list, valid_mask_list = [], []
    original_sizes_list, reshaped_sizes_list = [], []
    img_ids = []

    for item in batch:
        n   = item["gt_masks"].shape[0]
        pad = max_n - n

        pixel_values_list.append(item["pixel_values"])
        original_sizes_list.append(item["original_sizes"])
        reshaped_sizes_list.append(item["reshaped_input_sizes"])
        img_ids.append(item["img_id"])

        input_boxes_list.append(F.pad(item["input_boxes"], (0, 0, 0, 0, 0, pad)))
        gt_masks_list.append(F.pad(item["gt_masks"],       (0, 0, 0, 0, 0, pad)))
        ignore_mask_list.append(item["ignore_mask"])

        valid = torch.zeros(max_n, dtype=torch.bool)
        valid[:n] = True
        valid_mask_list.append(valid)

    return {
        "pixel_values":         torch.stack(pixel_values_list),
        "input_boxes":          torch.stack(input_boxes_list),
        "gt_masks":             torch.stack(gt_masks_list),
        "ignore_mask":          torch.stack(ignore_mask_list),
        "valid_mask":           torch.stack(valid_mask_list),
        "original_sizes":       torch.stack(original_sizes_list),
        "reshaped_input_sizes": torch.stack(reshaped_sizes_list),
        "img_ids":              img_ids,
    }


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def apply_lora(model, lora_r, lora_alpha, lora_dropout, target_modules):
    """Wrap the mask_decoder's attention layers with LoRA adapters.

    ``target_modules`` is passed directly to LoraConfig, so it can be either:
      - A single regex string (e.g. the default) that matches only
        mask_decoder attention projections.
      - A list of plain module-name suffixes if you want to target by type.

    Everything outside the matched modules stays frozen because PEFT only
    inserts trainable LoRA parameters for matched layers and leaves the rest
    of the model untouched (requires_grad=False by default for non-LoRA params
    when using get_peft_model).
    """
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, loss_type, epoch):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [train]")
    for batch in pbar:
        pv     = batch["pixel_values"].to(device)            # [B, C, H, W]
        ib     = batch["input_boxes"].squeeze(2).to(device)  # [B, N, 4]
        gt     = batch["gt_masks"].to(device)                # [B, N, 256, 256]
        ignore = batch["ignore_mask"].to(device)             # [B, 1, 256, 256]
        vm     = batch["valid_mask"].to(device)              # [B, N]

        out  = model(pixel_values=pv, input_boxes=ib, multimask_output=False)
        pred = out.pred_masks[:, :, 0, :, :]                 # [B, N, 256, 256]

        # Zero out loss for crowd-ignore regions AND padded slots
        valid = (1.0 - ignore) * vm[:, :, None, None].float()
        loss = seg_loss(pred, gt, loss_type, valid=valid)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, processor, device, coco_gt,
             output_dir, epoch_label, qual_images=5):
    model.eval()
    results    = []
    qual_count = 0
    qual_dir   = os.path.join(output_dir, "qualitative", epoch_label)
    os.makedirs(qual_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc=f"[{epoch_label}] eval"):
        item    = dataset[idx]
        img_id  = item["img_id"]
        cat_ids = item["cat_ids"].tolist()

        pv  = item["pixel_values"].unsqueeze(0).to(device)
        ib  = item["input_boxes"].squeeze(1).unsqueeze(0).to(device)
        os_ = item["original_sizes"].unsqueeze(0)
        ris = item["reshaped_input_sizes"].unsqueeze(0)

        out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)

        masks_list = processor.post_process_masks(
            masks=out.pred_masks,
            original_sizes=os_,
            reshaped_input_sizes=ris,
        )
        pred_masks = masks_list[0].cpu().float()

        for k, cat_id in enumerate(cat_ids):
            mask_np = (pred_masks[k, 0].numpy() > 0).astype(np.uint8)
            results.append({
                "image_id":     img_id,
                "category_id":  cat_id,
                "segmentation": mask_to_rle(mask_np),
                "score":        1.0,
            })

        if qual_count < qual_images:
            img_info = dataset.coco.loadImgs([img_id])[0]
            orig_img = Image.open(os.path.join(dataset.image_root, img_info["file_name"])).convert("RGB")
            pred_bin = [(pred_masks[k, 0].numpy() > 0) for k in range(len(cat_ids))]
            pred_vis = overlay_masks(orig_img, pred_bin, cat_ids)

            gt_anns     = dataset.coco.loadAnns(
                dataset.coco.getAnnIds(imgIds=[img_id], catIds=list(VALID_CAT_IDS)))
            gt_masks_np = [dataset.coco.annToMask(a) for a in gt_anns]
            gt_cat_ids  = [a["category_id"] for a in gt_anns]
            gt_vis      = overlay_masks(orig_img, gt_masks_np, gt_cat_ids)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].imshow(gt_vis);   axes[0].set_title("GT");        axes[0].axis("off")
            axes[1].imshow(pred_vis); axes[1].set_title("Predicted"); axes[1].axis("off")
            legend = [mpatches.Patch(color=(0,1,0), label="Pedestrian"),
                      mpatches.Patch(color=(0,0,1), label="Car")]
            fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=11)
            plt.tight_layout()
            seq_id, frame_id = img_id // 100000, img_id % 100000
            plt.savefig(os.path.join(qual_dir, f"seq{seq_id:04d}_frame{frame_id:06d}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close()
            qual_count += 1

    if not results:
        print("WARNING: no predictions generated.")
        return {}

    pred_coco = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, pred_coco, "segm")
    ev.params.imgIds = list({r["image_id"] for r in results})
    ev.params.catIds = list(VALID_CAT_IDS)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    metric_names = [
        "mAP_0.50_0.95", "mAP_0.50", "mAP_0.75",
        "mAP_small", "mAP_medium", "mAP_large",
        "mAR_1", "mAR_10", "mAR_100",
        "mAR_small", "mAR_medium", "mAR_large",
    ]
    metrics = {n: round(float(v), 4) for n, v in zip(metric_names, ev.stats)}

    precision = ev.eval["precision"]
    recall    = ev.eval["recall"]
    for k, cat_id in enumerate(sorted(VALID_CAT_IDS)):
        name = CAT_NAMES[cat_id]
        p = precision[:, :, k, 0, 2]
        r = recall[:, k, 0, 2]
        metrics[f"mAP_{name}"] = round(float(np.mean(p[p > -1])) if np.any(p > -1) else 0.0, 4)
        metrics[f"mAR_{name}"] = round(float(np.mean(r[r > -1])) if np.any(r > -1) else 0.0, 4)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune SAM mask-decoder with LoRA (peft)"
    )
    p.add_argument("--model_id",     default="facebook/sam-vit-base")

    # LoRA hyperparameters
    p.add_argument("--lora_r",       type=int,   default=8,
                   help="LoRA rank (default: 8)")
    p.add_argument("--lora_alpha",   type=float, default=16.0,
                   help="LoRA scaling factor (default: 16.0 = 2*r)")
    p.add_argument("--lora_dropout", type=float, default=0.1,
                   help="Dropout applied inside LoRA adapters (default: 0.1)")
    # A regex matching only the mask-decoder's attention projections.
    # Passed verbatim to LoraConfig(target_modules=...).
    # Users can override with a comma-separated list of plain module names
    # (e.g. "q_proj,v_proj") or another regex string.
    p.add_argument("--lora_target_modules",
                   default=r"mask_decoder\.transformer\..*\.(q_proj|v_proj)",
                   help=(
                       "Regex or comma-separated module names for LoRA targets. "
                       "Default targets only query and value projections inside "
                       "mask_decoder.transformer (standard LoRA practice)."
                   ))

    p.add_argument("--loss",         default="focal_dice", choices=["focal_dice", "bce"])
    p.add_argument("--augment",      action="store_true")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int,   default=1,
                   help="Epochs of linear LR warmup before cosine decay (default: 1)")
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--qual_images",  type=int,   default=5)
    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week2/outputs/task_e_lora")
    # wandb
    p.add_argument("--wandb_project", default="kitti-mots-sam-finetuning")
    p.add_argument("--wandb_entity",  default="just-an-arbitrary-team-name")
    p.add_argument("--wandb_run_name", default=None,
                   help="Override the auto-generated run name")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If the user passed a comma-separated list instead of a regex, convert it.
    if "," in args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    else:
        target_modules = args.lora_target_modules  # regex string

    aug_tag   = "aug" if args.augment else "noaug"
    model_tag = args.model_id.split("/")[-1]
    run_name  = f"lora_r{args.lora_r}_{args.loss}_{aug_tag}_{model_tag}"
    out_dir   = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Device: {device}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or run_name,
            config={
                "model_id":            args.model_id,
                "lora_r":              args.lora_r,
                "lora_alpha":          args.lora_alpha,
                "lora_dropout":        args.lora_dropout,
                "lora_target_modules": str(target_modules),
                "loss":                args.loss,
                "augment":             args.augment,
                "epochs":              args.epochs,
                "batch_size":          args.batch_size,
                "lr":                  args.lr,
                "warmup_epochs":       args.warmup_epochs,
                "weight_decay":        args.weight_decay,
            },
        )

    print("Loading annotations...")
    coco, fixed_path = build_coco_fixed(args.ann_file, out_dir)

    print("Building datasets...")
    processor = SamProcessor.from_pretrained(args.model_id)
    train_ds  = KITTISAMDataset(coco, args.dataset_root, TRAIN_SEQS, processor, augment=args.augment)
    val_ds    = KITTISAMDataset(coco, args.dataset_root, VAL_SEQS,   processor, augment=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )

    print(f"Loading {args.model_id}...")
    model = SamModel.from_pretrained(args.model_id).to(device)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, target_modules)

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

    history    = {"args": vars(args), "lora_target_modules": str(target_modules), "epochs": []}
    best_map   = 0.0
    best_epoch = 0
    best_adapter_dir = os.path.join(out_dir, "best_adapter")

    print("\n=== Epoch 0 / pretrained baseline ===")
    init_metrics = evaluate(model, val_ds, processor, device, coco,
                            out_dir, "epoch00", qual_images=args.qual_images)
    init_map = init_metrics.get("mAP_0.50_0.95", 0.0)
    print(f"  mAP[.5:.95]={init_map:.4f}  mAP@.50={init_metrics.get('mAP_0.50', 0):.4f}")
    print(f"  Pedestrian={init_metrics.get('mAP_Pedestrian', 0):.4f}"
          f"  Car={init_metrics.get('mAP_Car', 0):.4f}")
    history["epochs"].append({"epoch": 0, "train_loss": None, "val": init_metrics})
    if not args.no_wandb:
        wandb.log({"epoch": 0, **{f"val/{k}": v for k, v in init_metrics.items()}})

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device,
                                     args.loss, epoch)
        scheduler.step()

        print(f"\n=== Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} ===")
        val_metrics = evaluate(model, val_ds, processor, device, coco,
                               out_dir, f"epoch{epoch+1:02d}", qual_images=args.qual_images)

        cur_map = val_metrics.get("mAP_0.50_0.95", 0.0)
        print(f"  mAP[.5:.95]={cur_map:.4f}  mAP@.50={val_metrics.get('mAP_0.50', 0):.4f}")
        print(f"  Pedestrian={val_metrics.get('mAP_Pedestrian', 0):.4f}"
              f"  Car={val_metrics.get('mAP_Car', 0):.4f}")

        history["epochs"].append({"epoch": epoch+1, "train_loss": train_loss, "val": val_metrics})

        if not args.no_wandb:
            wandb.log({
                "epoch":      epoch + 1,
                "train/loss": train_loss,
                "train/lr":   scheduler.get_last_lr()[0],
                **{f"val/{k}": v for k, v in val_metrics.items()},
            })

        # Save LoRA adapter (lightweight checkpoint, ~MBs vs ~GBs for full weights)
        adapter_ckpt_dir = os.path.join(out_dir, f"adapter_ep{epoch+1:02d}")
        model.save_pretrained(adapter_ckpt_dir)
        print(f"  Adapter saved → {adapter_ckpt_dir}")

        if cur_map > best_map:
            best_map, best_epoch = cur_map, epoch + 1
            model.save_pretrained(best_adapter_dir)
            print(f"  *** New best: mAP={best_map:.4f}  adapter → {best_adapter_dir}")

    history["best_epoch"] = best_epoch
    history["best_map"]   = best_map
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    # Save merged checkpoints (adapter weights baked into model weights)
    # ------------------------------------------------------------------

    # 1. Last-epoch merged model
    print("\nMerging last-epoch LoRA weights into base model...")
    last_merged = model.merge_and_unload()
    last_merged_path = os.path.join(out_dir, "last_model_merged.pt")
    torch.save(last_merged.state_dict(), last_merged_path)
    print(f"  Merged (last) → {last_merged_path}")

    # 2. Best-epoch merged model (reload adapter, merge, save)
    print(f"Loading best adapter (epoch {best_epoch}) and merging...")
    best_base   = SamModel.from_pretrained(args.model_id).to(device)
    best_peft   = PeftModel.from_pretrained(best_base, best_adapter_dir)
    best_merged = best_peft.merge_and_unload()
    best_merged_path = os.path.join(out_dir, "best_model_merged.pt")
    torch.save(best_merged.state_dict(), best_merged_path)
    print(f"  Merged (best)  → {best_merged_path}")

    if not args.no_wandb:
        wandb.summary["best_epoch"]          = best_epoch
        wandb.summary["best_mAP_0.50_0.95"]  = best_map
        wandb.finish()

    print(f"\nDone. Best mAP={best_map:.4f} at epoch {best_epoch}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
