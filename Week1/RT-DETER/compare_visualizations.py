"""
Combine pretrained and finetuned RT-DETR visualizations side by side.
Reads existing images from both visualization folders — no GPU needed.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "pretrained_eval", "visualizations")
FINETUNED_DIR  = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "finetune_fixed3", "visualizations")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "Results_RTDETR", "comparison")

LABEL_HEIGHT = 40   # pixels reserved for the text banner above each image
FONT_SIZE    = 24
GAP          = 8    # horizontal gap between the two images


def make_banner(width, height, text, bg_color=(30, 30, 30), fg_color=(255, 255, 255)):
    banner = Image.new("RGB", (width, height), bg_color)
    draw   = ImageDraw.Draw(banner)
    try:
        draw.text((width // 2, height // 2), text, fill=fg_color,
                  font_size=FONT_SIZE, anchor="mm")
    except TypeError:
        # Older Pillow without font_size kwarg
        draw.text((10, 8), text, fill=fg_color)
    return banner


def combine(pretrained_path: Path, finetuned_path: Path, seq_id: str) -> Image.Image:
    left  = Image.open(pretrained_path).convert("RGB")
    right = Image.open(finetuned_path).convert("RGB")

    # Resize right to match left height (in case of different resolutions)
    if left.height != right.height:
        right = right.resize(
            (int(right.width * left.height / right.height), left.height),
            Image.LANCZOS,
        )

    w_left, h = left.size
    w_right   = right.width

    banner_left  = make_banner(w_left,  LABEL_HEIGHT, "Pretrained (COCO)")
    banner_right = make_banner(w_right, LABEL_HEIGHT, "Finetuned (fixed3)", bg_color=(20, 60, 20))

    total_w = w_left + GAP + w_right
    total_h = LABEL_HEIGHT + h

    canvas = Image.new("RGB", (total_w, total_h), (50, 50, 50))
    canvas.paste(banner_left,  (0,             0))
    canvas.paste(banner_right, (w_left + GAP,  0))
    canvas.paste(left,         (0,             LABEL_HEIGHT))
    canvas.paste(right,        (w_left + GAP,  LABEL_HEIGHT))

    return canvas


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pretrained_imgs = {p.name.replace("_pretrained", ""): p
                       for p in sorted(Path(PRETRAINED_DIR).glob("*.png"))}
    finetuned_imgs  = {p.name.replace("_finetuned",  ""): p
                       for p in sorted(Path(FINETUNED_DIR).glob("*.png"))}

    common = sorted(pretrained_imgs.keys() & finetuned_imgs.keys())
    if not common:
        print("No matching sequences found in both folders.")
        return

    print(f"Combining {len(common)} sequences...")
    for key in common:
        seq_id = key.replace(".png", "")
        out = combine(pretrained_imgs[key], finetuned_imgs[key], seq_id)
        save_path = os.path.join(OUTPUT_DIR, f"{seq_id}_comparison.png")
        out.save(save_path)
        print(f"  Saved: {save_path}")

    print(f"\nDone. {len(common)} comparison images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
