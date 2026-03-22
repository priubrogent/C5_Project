"""
qualitative_eval.py — Generate captioning examples for slides.

Loads a trained model from a run directory, runs inference on a random
sample of val and/or test images, and saves:
  - Individual annotated PNGs (image + predicted caption + GT captions)
  - A combined grid PNG ready to drop into slides
  - A JSON with all predictions for reference

Usage:
    python qualitative_eval.py --run_dir outputs/resnet50_gru_subword_adamw_1e3
    python qualitative_eval.py --run_dir outputs/resnet50_gru_subword_adamw_1e3 \\
        --splits val test --n_samples 12 --cols 4 --checkpoint best_loss_model.pt
"""

import argparse
import json
import os
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from dataset import VizWizDataset
from models import CaptioningModel
from tokenizer import build_tokenizer

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ── helpers ──────────────────────────────────────────────────────────────────

def denorm(tensor):
    """Reverse ImageNet normalisation → PIL image."""
    img = tensor.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return to_pil_image(img.clamp(0, 1))


def wrap(text, width=38):
    return "\n".join(textwrap.wrap(text, width))


@torch.no_grad()
def run_inference(model, dataset, indices, tokenizer, device):
    model.eval()
    results = []
    for idx in indices:
        img_tensor, _, gt_captions = dataset[idx]
        fname = dataset.samples[idx][0]
        gen = model.generate(
            img_tensor.unsqueeze(0).to(device),
            tokenizer.max_len - 1,
            tokenizer.sos_idx,
            tokenizer.eos_idx,
        )
        pred = tokenizer.decode(gen[0].cpu().tolist())
        results.append({
            "idx":        idx,
            "fname":      fname,
            "image_path": os.path.join(dataset.img_dir, fname),
            "prediction": pred,
            "gt_captions": gt_captions,
            "pil_image":  denorm(img_tensor),
        })
    return results


# ── plotting ─────────────────────────────────────────────────────────────────

def make_panel(ax, result, show_gt=True, max_gt=2):
    """Draw one image panel with caption text."""
    ax.imshow(result["pil_image"])
    ax.axis("off")

    pred_text = f"Pred: {wrap(result['prediction'])}"
    ax.text(
        0.5, -0.02, pred_text,
        transform=ax.transAxes,
        fontsize=7.5, color="#1a1a2e",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#d0e8ff", alpha=0.85, linewidth=0),
    )

    if show_gt and result["gt_captions"]:
        gt_lines = [f"GT{i+1}: {wrap(c, 36)}"
                    for i, c in enumerate(result["gt_captions"][:max_gt])]
        gt_text = "\n".join(gt_lines)
        ax.text(
            0.5, -0.22, gt_text,
            transform=ax.transAxes,
            fontsize=6.5, color="#2d2d2d",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.85, linewidth=0),
        )


def save_grid(results, out_path, cols=4, show_gt=True, title=None):
    n    = len(results)
    rows = (n + cols - 1) // cols
    h_per_row = 3.5 if show_gt else 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * h_per_row))
    axes = axes.flatten() if n > 1 else [axes]

    for i, result in enumerate(results):
        make_panel(axes[i], result, show_gt=show_gt)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid  → {out_path}")


def save_individuals(results, out_dir, show_gt=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        fig, ax = plt.subplots(figsize=(4, 4.5 if show_gt else 3.5))
        make_panel(ax, r, show_gt=show_gt)
        fig.tight_layout()
        stem = Path(r["fname"]).stem
        p = out_dir / f"{stem}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {len(results)} individual images → {out_dir}/")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir",    required=True,
                   help="Path to the run output dir (must contain config.json)")
    p.add_argument("--checkpoint", default="best_metric_model.pt",
                   help="Checkpoint filename inside run_dir (default: best_metric_model.pt)")
    p.add_argument("--data_root",  default=None,
                   help="Dataset root. Defaults to the value stored in config.json")
    p.add_argument("--splits",     nargs="+", default=["val", "test"],
                   choices=["val", "test"])
    p.add_argument("--n_samples",  type=int, default=16,
                   help="Number of images to sample per split")
    p.add_argument("--cols",       type=int, default=4,
                   help="Columns in the output grid")
    p.add_argument("--no_gt",      action="store_true",
                   help="Hide GT captions (cleaner for slides)")
    p.add_argument("--out_dir",    default=None,
                   help="Where to save results. Defaults to run_dir/qualitative/")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    cfg     = json.loads((run_dir / "config.json").read_text())

    data_root = args.data_root or cfg["data_root"]
    out_dir   = Path(args.out_dir) if args.out_dir else run_dir / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Run     : {run_dir.name}")
    print(f"Ckpt    : {args.checkpoint}")
    print(f"Device  : {device}")
    print(f"Splits  : {args.splits}  n_samples={args.n_samples}")

    # ── tokenizer & model ────────────────────────────────────────────────────
    cache_dir = os.path.join(data_root, "tokenizer_cache")
    train_ann = os.path.join(data_root, "annotations", "train.json")
    tokenizer = build_tokenizer(cfg["text_repr"], train_ann, cache_dir,
                                max_len=cfg.get("max_len"))

    model = CaptioningModel(
        encoder_name  = cfg["encoder"],
        decoder_type  = cfg["decoder"],
        decoder_layers= cfg["decoder_layers"],
        vocab_size    = tokenizer.vocab_size,
        embed_dim     = cfg["embed_dim"],
        hidden_dim    = cfg["hidden_dim"],
        dropout       = cfg.get("dropout", 0.0),
    ).to(device)

    ckpt_path = run_dir / args.checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : {cfg['encoder']} + {cfg['decoder']}×{cfg['decoder_layers']} "
          f"| {cfg['text_repr']} | {n_params/1e6:.1f}M params")

    # ── per-split inference ───────────────────────────────────────────────────
    show_gt = not args.no_gt
    all_records = {}

    for split in args.splits:
        print(f"\n── {split} split ──────────────────────────────────────")
        if split == "val":
            ds = VizWizDataset(
                os.path.join(data_root, "train"),
                os.path.join(data_root, "annotations", "train.json"),
                tokenizer, split="val",
                val_fraction=cfg.get("val_fraction", 0.1),
                seed=cfg.get("seed", 42),
            )
        else:
            ds = VizWizDataset(
                os.path.join(data_root, "val"),
                os.path.join(data_root, "annotations", "val.json"),
                tokenizer, split="test",
                seed=cfg.get("seed", 42),
            )
        print(f"  Dataset size: {len(ds)}")

        n = min(args.n_samples, len(ds))
        indices = random.sample(range(len(ds)), n)
        results = run_inference(model, ds, indices, tokenizer, device)

        # print predictions to terminal
        for r in results:
            gt_preview = r["gt_captions"][0] if r["gt_captions"] else "N/A"
            print(f"  [{r['fname']}]")
            print(f"    Pred : {r['prediction']}")
            print(f"    GT1  : {gt_preview}")

        # grid PNG
        grid_title = f"{run_dir.name}  |  {split} set  ({n} samples)"
        save_grid(results, out_dir / f"grid_{split}.png",
                  cols=args.cols, show_gt=show_gt, title=grid_title)

        # individual PNGs
        save_individuals(results, out_dir / f"individual_{split}", show_gt=show_gt)

        # JSON record (strip pil_image before saving)
        records = [{k: v for k, v in r.items() if k != "pil_image"} for r in results]
        all_records[split] = records

    # combined JSON
    json_path = out_dir / "predictions.json"
    json_path.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))
    print(f"\n  Saved JSON  → {json_path}")
    print(f"\nDone. All outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
