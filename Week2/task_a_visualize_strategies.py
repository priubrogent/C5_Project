import os
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

TRAIN_SEQS = [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]
VAL_SEQS   = [2, 6, 7, 8, 10, 13, 14, 16, 18]

VALID_CAT_IDS  = {1, 3}
CAT_NAMES      = {1: "Pedestrian", 3: "Car"}
BBOX_COLORS    = {1: "lime", 3: "cyan"}
MASK_STRATEGIES = {"mask_centroid", "random_mask"}

STRATEGIES = [
    ("bbox_center",   1),
    ("mask_centroid", 1),
    ("random_mask",   1),
    ("random_mask",   3),
    ("random_bbox",   1),
    ("random_bbox",   3),
]


def decode_gt_mask(ann, img_info):
    segm = ann["segmentation"]
    h, w = img_info["height"], img_info["width"]
    if isinstance(segm, dict):
        rle = segm
    else:
        rle = {"counts": segm, "size": [h, w]}
    return mask_util.decode(rle).astype(np.uint8)


def build_coco_gt_segm(path):
    with open(path) as f:
        data = json.load(f)
    img_size = {img["id"]: (img["height"], img["width"]) for img in data["images"]}
    for ann in data["annotations"]:
        if not isinstance(ann["segmentation"], dict):
            h, w = img_size[ann["image_id"]]
            ann["segmentation"] = {"counts": ann["segmentation"], "size": [h, w]}
    return data


def get_point_prompts(ann, img_info, strategy, num_points, rng):
    x, y, w, h = ann["bbox"]
    if strategy == "bbox_center":
        return [[x + w / 2, y + h / 2]], [1]
    if strategy == "mask_centroid":
        mask = decode_gt_mask(ann, img_info)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [[x + w / 2, y + h / 2]], [1]
        return [[float(np.mean(xs)), float(np.mean(ys))]], [1]
    if strategy == "random_mask":
        mask = decode_gt_mask(ann, img_info)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            pts = [[x + w / 2, y + h / 2]] * num_points
        else:
            idxs = rng.choice(len(xs), size=min(num_points, len(xs)), replace=False)
            pts = [[float(xs[i]), float(ys[i])] for i in idxs]
            while len(pts) < num_points:
                pts.append(pts[-1])
        return pts, [1] * len(pts)
    if strategy == "random_bbox":
        pts = [[float(rng.uniform(x, x + w)), float(rng.uniform(y, y + h))]
               for _ in range(num_points)]
        return pts, [1] * len(pts)
    raise ValueError(f"Unknown strategy: {strategy}")


def draw_panel(ax, image, anns, img_info, strategy, num_points, seed):
    ax.imshow(image)
    use_mask = strategy in MASK_STRATEGIES
    for ann in anns:
        color = BBOX_COLORS.get(ann["category_id"], "white")
        if use_mask:
            mask = decode_gt_mask(ann, img_info)
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=[1])
        else:
            bx, by, bw, bh = ann["bbox"]
            ax.add_patch(patches.Rectangle(
                (bx, by), bw, bh, linewidth=1, edgecolor=color, facecolor="none"
            ))
        pts, _ = get_point_prompts(ann, img_info, strategy, num_points,
                                   np.random.default_rng(seed))
        for px, py in pts:
            ax.plot(px, py, "r+", markersize=8, markeredgewidth=1.5)
    title = strategy if strategy not in ("random_mask", "random_bbox") else f"{strategy}  n={num_points}"
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--dataset_root",
                   default="/home/arnau-marcos-almansa/Downloads/KITTI-MOTS/training/image_02")
    p.add_argument("--ann_file",
                   default="/home/arnau-marcos-almansa/workspace/C5_Project/Week1/kitti_mots_to_coco_gt.json")

    p.add_argument("--output_dir", default="./outputs/task_a_visualize")
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--image_path", default=None,
                   help="Substring of file_name to match (e.g. '0002/000042')")
    p.add_argument("--image_index", type=int, default=0,
                   help="Index into the filtered image list")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gt_data = build_coco_gt_segm(args.ann_file)
    fixed_gt_path = os.path.join(args.output_dir, "gt_fixed.json")
    with open(fixed_gt_path, "w") as f:
        json.dump(gt_data, f)
    coco_gt = COCO(fixed_gt_path)

    if args.split == "val":
        target_seqs = VAL_SEQS
    elif args.split == "train":
        target_seqs = TRAIN_SEQS
    else:
        target_seqs = TRAIN_SEQS + VAL_SEQS

    all_img_ids = coco_gt.getImgIds()
    img_ids = [iid for iid in all_img_ids if (iid // 100000) in target_seqs]
    img_ids_filtered = [
        iid for iid in img_ids
        if os.path.exists(os.path.join(args.dataset_root,
                                       coco_gt.loadImgs([iid])[0]["file_name"]))
    ]

    if args.image_path:
        matched = [iid for iid in img_ids_filtered
                   if args.image_path in coco_gt.loadImgs([iid])[0]["file_name"]]
        if not matched:
            raise ValueError(f"No image found matching: {args.image_path}")
        img_id = matched[0]
    else:
        img_id = img_ids_filtered[args.image_index]

    img_info = coco_gt.loadImgs([img_id])[0]
    img_path = os.path.join(args.dataset_root, img_info["file_name"])
    image = Image.open(img_path).convert("RGB")

    ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
    anns = coco_gt.loadAnns(ann_ids)
    anns = [a for a in anns if a["category_id"] in VALID_CAT_IDS and a.get("iscrowd", 0) == 0]

    if not anns:
        print("No valid annotations in this image.")
        return

    seq_id   = img_id // 100000
    frame_id = img_id % 100000

    for strategy, num_points in STRATEGIES:
        fig, ax = plt.subplots(figsize=(12, 5))
        draw_panel(ax, image, anns, img_info, strategy, num_points, args.seed)

        use_mask = strategy in MASK_STRATEGIES
        contour_line = plt.Line2D([0], [0], color="lime", linewidth=1, label="Pedestrian mask edge")
        bbox_patch    = patches.Patch(edgecolor="lime", facecolor="none", label="Pedestrian bbox")
        legend_handles = [
            contour_line if use_mask else bbox_patch,
            plt.Line2D([0], [0], color="cyan", linewidth=1, label="Car mask edge")
            if use_mask else
            patches.Patch(edgecolor="cyan", facecolor="none", label="Car bbox"),
            plt.Line2D([0], [0], marker="+", color="red", linestyle="none",
                       markersize=10, markeredgewidth=2, label="Selected point"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10)
        fig.suptitle(f"seq{seq_id:04d}_frame{frame_id:06d}", fontsize=11)
        plt.tight_layout(rect=[0, 0.06, 1, 1])

        suffix = strategy if strategy not in ("random_mask", "random_bbox") else f"{strategy}_n{num_points}"
        out_path = os.path.join(args.output_dir,
                                f"{suffix}_seq{seq_id:04d}_frame{frame_id:06d}.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"Saved to {out_path}")

        if not args.no_show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
