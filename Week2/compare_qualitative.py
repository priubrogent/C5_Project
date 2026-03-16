"""
Compare prediction panels (rightmost third) of 3-panel qualitative images
from two directories and save side-by-side comparisons ranked by difference.

Output per frame: [label_a pred] | [label_b pred] | [diff map: A in red where different]

Usage:
    python compare_qualitative.py DIR_A DIR_B \
        --label_a "YOLO+SAM" --label_b "DINO-tiny+SAM-large" \
        --output_dir outputs/comparison \
        --top 15 \
        --diff_threshold 20
"""

import argparse
import os
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a", help="First qualitative directory")
    p.add_argument("dir_b", help="Second qualitative directory")
    p.add_argument("--label_a", default="Method A", help="Label for dir_a")
    p.add_argument("--label_b", default="Method B", help="Label for dir_b")
    p.add_argument("--output_dir", default="outputs/qual_comparison")
    p.add_argument("--top", type=int, default=15, help="How many most-different frames to save")
    p.add_argument("--diff_threshold", type=int, default=20,
                   help="Per-channel mean diff above which a pixel is considered different (0-255)")
    return p.parse_args()


def get_pred_panel(img: Image.Image) -> Image.Image:
    """Extract the rightmost third (prediction panel) from a 3-panel image."""
    w, h = img.size
    return img.crop((2 * w // 3, 0, w, h))


def make_diff_map(arr_a: np.ndarray, arr_b_resized: np.ndarray, threshold: int) -> np.ndarray:
    """Return arr_a with pixels that differ from arr_b painted solid red."""
    diff = np.mean(np.abs(arr_a.astype(float) - arr_b_resized.astype(float)), axis=2)
    result = arr_a.copy()
    mask = diff > threshold
    result[mask] = [255, 0, 0]
    return result


def pad_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Pad image at the bottom with black to reach target_h."""
    if img.height == target_h:
        return img
    canvas = Image.new("RGB", (img.width, target_h), (0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas


def main():
    args = parse_args()

    files_a = set(os.listdir(args.dir_a))
    files_b = set(os.listdir(args.dir_b))
    common = sorted(files_a & files_b)
    print(f"Found {len(common)} common frames between:\n  A: {args.dir_a}\n  B: {args.dir_b}")

    if not common:
        print("No common files found — check the directories.")
        return

    # Score each frame by mean absolute pixel difference between prediction panels
    scores = []
    for fname in common:
        img_a = Image.open(os.path.join(args.dir_a, fname)).convert("RGB")
        img_b = Image.open(os.path.join(args.dir_b, fname)).convert("RGB")

        arr_a = np.array(get_pred_panel(img_a)).astype(float)
        arr_b = np.array(
            get_pred_panel(img_b).resize((arr_a.shape[1], arr_a.shape[0]))
        ).astype(float)

        diff = np.mean(np.abs(arr_a - arr_b))
        scores.append((diff, fname))

    scores.sort(reverse=True)

    print(f"\nTop {args.top} most different frames:")
    for diff, fname in scores[:args.top]:
        print(f"  {fname}  diff={diff:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)
    label_h = 20
    gap = 6

    for diff, fname in scores[:args.top]:
        img_a = Image.open(os.path.join(args.dir_a, fname)).convert("RGB")
        img_b = Image.open(os.path.join(args.dir_b, fname)).convert("RGB")

        pred_a = get_pred_panel(img_a)
        pred_b = get_pred_panel(img_b)

        # Resize B to A's size only for diff computation
        arr_a = np.array(pred_a)
        arr_b_for_diff = np.array(pred_b.resize((pred_a.width, pred_a.height)))
        diff_map = Image.fromarray(make_diff_map(arr_a, arr_b_for_diff, args.diff_threshold))

        # Pad all three panels to the same height (tallest of the three)
        max_h = max(pred_a.height, pred_b.height, diff_map.height)
        pred_a    = pad_to_height(pred_a,   max_h)
        pred_b    = pad_to_height(pred_b,   max_h)
        diff_map  = pad_to_height(diff_map, max_h)

        total_w = pred_a.width + gap + pred_b.width + gap + diff_map.width
        canvas = Image.new("RGB", (total_w, max_h + label_h), (30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        x = 0
        canvas.paste(pred_a,   (x, label_h)); draw.text((x + 2, 2), args.label_a,               fill=(255, 255, 100))
        x += pred_a.width + gap
        canvas.paste(pred_b,   (x, label_h)); draw.text((x + 2, 2), args.label_b,               fill=(100, 200, 255))
        x += pred_b.width + gap
        canvas.paste(diff_map, (x, label_h)); draw.text((x + 2, 2), f"diff (A, thr={args.diff_threshold})", fill=(255, 100, 100))

        canvas.save(os.path.join(args.output_dir, fname))

    print(f"\nComparison images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
