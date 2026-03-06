import os
import json
import argparse
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from transformers import SamModel, SamProcessor

VALID_CAT_IDS = {1, 3}
CAT_NAMES = {1: "Pedestrian", 3: "Car"}


def decode_gt_mask(ann, img_info):
    segm = ann["segmentation"]
    h, w = img_info["height"], img_info["width"]
    if isinstance(segm, dict):
        rle = segm
    else:
        rle = {"counts": segm, "size": [h, w]}
    return mask_util.decode(rle).astype(np.uint8)


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def overlay(image_np, mask, color, alpha=0.5):
    out = image_np.copy().astype(float)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + v * alpha, out[:, :, c])
    return out.astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="facebook/sam-vit-base")
    p.add_argument("--dataset_root",
                   default="/hhome/priubrogent/mcv/datasets/C5/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/hhome/priubrogent/mcvpol/C5/Week1/R-CNN/kitti_mots_to_coco_gt.json")
    p.add_argument("--output_dir",
                   default="/hhome/priubrogent/mcvpol/C5/Week2/outputs/single_inference")
    p.add_argument("--strategy",
                   choices=["bbox_center", "mask_centroid", "random_mask", "random_bbox"],
                   default="bbox_center")
    p.add_argument("--num_points", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model: {args.model}")

    processor = SamProcessor.from_pretrained(args.model)
    model = SamModel.from_pretrained(args.model).to(device)
    model.eval()

    with open(args.ann_file) as f:
        raw = json.load(f)

    img_size = {img["id"]: (img["height"], img["width"]) for img in raw["images"]}
    for ann in raw["annotations"]:
        if not isinstance(ann["segmentation"], dict):
            h, w = img_size[ann["image_id"]]
            ann["segmentation"] = {"counts": ann["segmentation"], "size": [h, w]}

    gt_path = os.path.join(args.output_dir, "_gt_tmp.json")
    with open(gt_path, "w") as f:
        json.dump(raw, f)
    coco_gt = COCO(gt_path)

    img_record = next(
        (img for img in raw["images"] if img["file_name"] == args.image), None
    )
    if img_record is None:
        raise ValueError(f"Image '{args.image}' not found in annotation file.")

    img_id   = img_record["id"]
    img_info = coco_gt.loadImgs([img_id])[0]
    img_path = os.path.join(args.dataset_root, img_info["file_name"])
    image    = Image.open(img_path).convert("RGB")
    image_np = np.array(image)

    ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
    anns    = coco_gt.loadAnns(ann_ids)
    anns    = [a for a in anns if a["category_id"] in VALID_CAT_IDS and a.get("iscrowd", 0) == 0]

    if not anns:
        print("No valid annotations for this image.")
        return

    print(f"Image: {args.image}  |  Annotations: {len(anns)}")

    rng = np.random.default_rng(args.seed)
    input_points, input_labels = [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if args.strategy == "bbox_center":
            pts = [[x + w / 2, y + h / 2]]
        elif args.strategy == "mask_centroid":
            gt_mask = decode_gt_mask(ann, img_info)
            ys, xs = np.where(gt_mask > 0)
            pts = [[float(np.mean(xs)), float(np.mean(ys))]] if len(xs) > 0 else [[x + w / 2, y + h / 2]]
        elif args.strategy == "random_mask":
            gt_mask = decode_gt_mask(ann, img_info)
            ys, xs = np.where(gt_mask > 0)
            if len(xs) == 0:
                pts = [[x + w / 2, y + h / 2]] * args.num_points
            else:
                idxs = rng.choice(len(xs), size=min(args.num_points, len(xs)), replace=False)
                pts = [[float(xs[i]), float(ys[i])] for i in idxs]
                while len(pts) < args.num_points:
                    pts.append(pts[-1])
        else:
            pts = [[float(rng.uniform(x, x + w)), float(rng.uniform(y, y + h))]
                   for _ in range(args.num_points)]
        input_points.append(pts)
        input_labels.append([1] * len(pts))

    inputs = processor(
        images=image,
        input_points=[input_points],
        input_labels=[input_labels],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_masks_list = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    iou_scores = outputs.iou_scores.cpu().numpy()[0]
    pred_masks = pred_masks_list[0]

    N = len(anns)
    fig, axes = plt.subplots(N, 4, figsize=(18, 4 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    COLORS = [(0, 200, 0), (220, 80, 80), (80, 80, 220), (200, 160, 0)]

    for k, ann in enumerate(anns):
        gt_mask = decode_gt_mask(ann, img_info)
        cat_name = CAT_NAMES.get(ann["category_id"], "?")

        bx, by, bw, bh = [int(v) for v in ann["bbox"]]
        H_img, W_img = image_np.shape[:2]
        pad = max(20, int(max(bw, bh) * 0.15))
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(W_img, bx + bw + pad), min(H_img, by + bh + pad)
        crop = image_np[y1:y2, x1:x2]

        ax = axes[k, 0]
        gt_vis = overlay(crop, gt_mask[y1:y2, x1:x2], COLORS[0])
        ax.imshow(gt_vis)
        for pt in input_points[k]:
            ax.plot(pt[0] - x1, pt[1] - y1, "*", color="yellow", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.set_title(f"GT  [{cat_name}]", fontsize=9, fontweight="bold")
        ax.axis("off")

        best_idx = int(iou_scores[k].argmax())
        for m in range(3):
            pred_mask = pred_masks[k, m].numpy().astype(np.uint8)
            actual_iou = mask_iou(gt_mask, pred_mask)
            sam_score  = iou_scores[k, m]
            pred_vis = overlay(crop, pred_mask[y1:y2, x1:x2], COLORS[m + 1])
            ax = axes[k, m + 1]
            ax.imshow(pred_vis)
            for pt in input_points[k]:
                ax.plot(pt[0] - x1, pt[1] - y1, "*", color="yellow", markersize=12,
                        markeredgecolor="black", markeredgewidth=0.5)
            star = " ★" if m == best_idx else ""
            ax.set_title(
                f"Mask {m}{star}\nactual IoU={actual_iou:.3f}  SAM score={sam_score:.3f}",
                fontsize=8,
                color="darkgreen" if m == best_idx else "black",
            )
            ax.axis("off")

    plt.suptitle(
        f"{args.image}  |  strategy={args.strategy}\n"
        f"★ = mask SAM would pick  |  "
        f"actual IoU = overlap with GT  |  SAM score = model confidence",
        fontsize=9,
    )
    plt.tight_layout()

    out_name = args.image.replace("/", "_").replace(".png", "") + f"_{args.strategy}.png"
    out_path = os.path.join(args.output_dir, out_name)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
