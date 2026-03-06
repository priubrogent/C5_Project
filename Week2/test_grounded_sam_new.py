import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
plt.ion()
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

IMAGE_PATH = "000036.png"
LABELS = ["a car.", "a person."]
THRESHOLD = 0.3
DETECTOR_ID = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID = "facebook/sam-vit-base"
OUTPUT_PATH = "test_grounded_sam_out.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

image = Image.open(IMAGE_PATH).convert("RGB")

detector = pipeline(model=DETECTOR_ID, task="zero-shot-object-detection", device=device)
detections = detector(image, candidate_labels=LABELS, threshold=THRESHOLD)
print(f"Found {len(detections)} detections")

boxes = [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]] for d in detections]
#draw the boxes on the image
image_cv2 = np.array(image)
image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
for box in boxes:
    cv2.rectangle(image_cv2, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)) 

sam = AutoModelForMaskGeneration.from_pretrained(SEGMENTER_ID).to(device)
processor = AutoProcessor.from_pretrained(SEGMENTER_ID)

inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(device)
outputs = sam(**inputs)
masks = processor.post_process_masks(
    masks=outputs.pred_masks,
    original_sizes=inputs.original_sizes,
    reshaped_input_sizes=inputs.reshaped_input_sizes
)[0]

# masks shape: (N, 3, H, W) - 3 candidates per box; pick best by IOU score
iou_scores = outputs.iou_scores[0]          # (N, 3)
best_idx = iou_scores.argmax(dim=1)         # (N,)
masks_tensor = masks.cpu().float()          # (N, 3, H, W)
masks = np.stack([
    masks_tensor[i, best_idx[i]].numpy() > 0
    for i in range(len(best_idx))
])

# --- VISUALIZE ---
img_np = np.array(image)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.imshow(img_np)

colors = plt.cm.tab10.colors
for i, (det, mask) in enumerate(zip(detections, masks)):
    color = colors[i % len(colors)]
    colored_mask = np.zeros((*mask.shape, 4), dtype=float)
    colored_mask[mask] = [*color[:3], 0.5]
    ax.imshow(colored_mask[0])
    b = det["box"]
    rect = plt.Rectangle((b["xmin"], b["ymin"]), b["xmax"]-b["xmin"], b["ymax"]-b["ymin"],
                          linewidth=2, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    ax.text(b["xmin"], b["ymin"] - 5, f'{det["label"]} {det["score"]:.2f}',
            color=color, fontsize=9, fontweight="bold")

ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
