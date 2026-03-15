"""
Inference-only baseline for task_e.

Loads a SAM model (pretrained or a merged fine-tuned .pt checkpoint),
runs it on the validation split with GT bounding-box prompts, and
reports the same COCO segmentation metrics as task_e.py / task_e_lora.py.

Use this to establish the zero-shot SAM baseline and compare it
against fine-tuned checkpoints.

Examples
--------
# Zero-shot pretrained baseline:
python task_e_inference.py

# Evaluate a merged fine-tuned checkpoint:
python task_e_inference.py --checkpoint path/to/best_model_merged.pt

# Evaluate a PEFT adapter checkpoint:
python task_e_inference.py --peft_adapter path/to/best_adapter/
"""

import os, json, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

from transformers import SamModel, SamProcessor
from tqdm import tqdm

VALID_CAT_IDS = {1, 3}
CAT_NAMES     = {1: "Pedestrian", 3: "Car"}
MASK_COLORS   = {1: (0, 255, 0), 3: (0, 0, 255)}
SAM_PRED_SIZE = 256
VAL_SEQS      = [2, 6, 7, 8, 10, 13, 14, 16, 18]


# ---------------------------------------------------------------------------
# Helpers
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
    return COCO(fixed_path)

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
# Dataset
# ---------------------------------------------------------------------------

class KITTISAMDataset(Dataset):

    def __init__(self, coco, image_root, seqs, processor):
        self.coco       = coco
        self.image_root = image_root
        self.processor  = processor

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
        image    = Image.open(
            os.path.join(self.image_root, img_info["file_name"])
        ).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=list(VALID_CAT_IDS))
        anns    = [a for a in self.coco.loadAnns(ann_ids)
                   if a.get("iscrowd", 0) == 0 and a["bbox"][2] >= 2 and a["bbox"][3] >= 2]

        if not anns:
            return self.__getitem__((idx + 1) % len(self))

        boxes   = [[a["bbox"][0], a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                   for a in anns]
        cat_ids = [a["category_id"] for a in anns]

        inputs = self.processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt",
        )

        return {
            "pixel_values":         inputs["pixel_values"].squeeze(0),
            "input_boxes":          inputs["input_boxes"].squeeze(0).reshape(-1, 1, 4),
            "original_sizes":       inputs["original_sizes"].squeeze(0),
            "reshaped_input_sizes": inputs["reshaped_input_sizes"].squeeze(0),
            "cat_ids":              torch.tensor(cat_ids, dtype=torch.long),
            "img_id":               img_id,
        }


# ---------------------------------------------------------------------------
# Inference + evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, dataset, processor, device, coco_gt, out_dir, qual_images=5):
    model.eval()
    results    = []
    qual_count = 0
    qual_dir   = os.path.join(out_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Inference"):
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
            orig_img = Image.open(
                os.path.join(dataset.image_root, img_info["file_name"])
            ).convert("RGB")
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
        description="Inference-only evaluation for task_e (SAM with GT box prompts)"
    )
    p.add_argument("--model_id",   default="facebook/sam-vit-base",
                   help="HuggingFace model ID for the base SAM architecture")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a merged fine-tuned .pt state-dict. "
                        "If omitted, uses the raw pretrained weights.")
    p.add_argument("--peft_adapter", default=None,
                   help="Path to a PEFT adapter directory (alternative to --checkpoint). "
                        "Requires peft to be installed.")
    p.add_argument("--qual_images", type=int, default=5)
    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week2/outputs/task_e_inference")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine run label for output directory
    if args.checkpoint:
        run_label = "finetuned_" + os.path.splitext(os.path.basename(args.checkpoint))[0]
    elif args.peft_adapter:
        run_label = "peft_" + os.path.basename(args.peft_adapter.rstrip("/"))
    else:
        run_label = "pretrained_" + args.model_id.split("/")[-1]

    out_dir = os.path.join(args.output_dir, run_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Run label : {run_label}")
    print(f"Device    : {device}")

    # Load model
    print(f"Loading base model {args.model_id}...")
    model = SamModel.from_pretrained(args.model_id)

    if args.checkpoint:
        print(f"Loading merged checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state)
    elif args.peft_adapter:
        from peft import PeftModel
        print(f"Loading PEFT adapter: {args.peft_adapter}")
        model = PeftModel.from_pretrained(model, args.peft_adapter)
        model = model.merge_and_unload()

    model = model.to(device)
    model.eval()

    processor = SamProcessor.from_pretrained(args.model_id)

    print("Loading annotations...")
    coco = build_coco_fixed(args.ann_file, out_dir)

    print("Building validation dataset...")
    val_ds = KITTISAMDataset(coco, args.dataset_root, VAL_SEQS, processor)
    print(f"Val: {len(val_ds)} images")

    metrics = run_inference(model, val_ds, processor, device, coco,
                            out_dir, qual_images=args.qual_images)

    print("\n=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    result = {"run_label": run_label, "args": vars(args), "metrics": metrics}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
