"""
Shared utilities for Task F dataset evaluation scripts.
"""

import numpy as np
from PIL import Image

import torch
import pycocotools.mask as mask_util
from transformers import SamModel, SamProcessor


def load_sam(model_id: str, checkpoint: str | None = None) -> tuple[SamModel, SamProcessor]:
    """Load a SAM model, optionally from a merged fine-tuned .pt checkpoint."""
    print(f"  Loading base model: {model_id}")
    model = SamModel.from_pretrained(model_id)
    if checkpoint is not None:
        print(f"  Loading checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
    processor = SamProcessor.from_pretrained(model_id)
    return model, processor


def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode a binary H×W mask as COCO RLE."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def overlay_masks(
    image: Image.Image,
    masks: list,
    colors: list,
    alpha: float = 0.45,
) -> Image.Image:
    """Alpha-blend a list of binary masks onto a PIL image.

    Args:
        masks:  list of H×W bool/uint8 arrays
        colors: list of (R, G, B) tuples (0-255) matching masks
    """
    img = np.array(image.convert("RGB")).copy().astype(float)
    for mask, color in zip(masks, colors):
        for c, v in enumerate(color):
            img[:, :, c] = np.where(mask, img[:, :, c] * (1 - alpha) + v * alpha, img[:, :, c])
    return Image.fromarray(img.clip(0, 255).astype(np.uint8))


@torch.no_grad()
def sam_predict_boxes(
    model: SamModel,
    processor: SamProcessor,
    image: Image.Image,
    boxes_xyxy: list,
    device: str,
) -> np.ndarray:
    """Run SAM on one image with GT bounding-box prompts.

    Args:
        boxes_xyxy: list of [x1, y1, x2, y2] in pixel coordinates

    Returns:
        masks: bool ndarray of shape (N, H, W)
    """
    inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt")
    pv  = inputs["pixel_values"].to(device)
    ib  = inputs["input_boxes"].reshape(1, -1, 4).to(device)  # (1, N, 4)
    os_ = inputs["original_sizes"]
    ris = inputs["reshaped_input_sizes"]

    out = model(pixel_values=pv, input_boxes=ib, multimask_output=False)

    masks_list = processor.post_process_masks(
        masks=out.pred_masks,
        original_sizes=os_,
        reshaped_input_sizes=ris,
    )
    # masks_list[0]: (N, 1, H, W)
    pred = masks_list[0].cpu().float()
    return (pred[:, 0].numpy() > 0)  # (N, H, W) bool
