import os
import glob
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True,
                   help="Directory with strategy images (output of task_a_visualize_strategies.py)")
    p.add_argument("--reference", default=None,
                   help="Substring to match the reference image. Defaults to 'bbox_center'.")
    p.add_argument("--pattern", default="*.png",
                   help="Glob pattern to match images to crop")
    p.add_argument("--output_dir", default=None,
                   help="Where to save cropped images. Defaults to input_dir/cropped/")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.input_dir, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    all_images = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not all_images:
        print(f"No images found matching {os.path.join(args.input_dir, args.pattern)}")
        return

    key = args.reference or "bbox_center"
    ref_candidates = [p for p in all_images if key in os.path.basename(p)]
    ref_path = ref_candidates[0] if ref_candidates else all_images[0]
    print(f"Reference: {ref_path}")
    print("Click the top-left and bottom-right corners of the crop area.")

    ref_img = Image.open(ref_path)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(ref_img)
    ax.set_title("Click two corners of the crop area (top-left, bottom-right)")
    ax.axis("off")
    plt.tight_layout()

    pts = plt.ginput(2, timeout=0)
    plt.close(fig)

    if len(pts) < 2:
        print("Selection cancelled.")
        return

    x0 = int(min(pts[0][0], pts[1][0]))
    y0 = int(min(pts[0][1], pts[1][1]))
    x1 = int(max(pts[0][0], pts[1][0]))
    y1 = int(max(pts[0][1], pts[1][1]))
    print(f"Crop: ({x0}, {y0}) -> ({x1}, {y1})")

    # Preview crop on reference
    preview = ref_img.crop((x0, y0, x1, y1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(preview)
    ax.set_title(f"Preview — close to apply to all {len(all_images)} images")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    for img_path in all_images:
        img = Image.open(img_path)
        cropped = img.crop((x0, y0, x1, y1))
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        cropped.save(out_path)
        print(f"Saved: {out_path}")

    print(f"\nDone. {len(all_images)} images saved to {output_dir}")


if __name__ == "__main__":
    main()
