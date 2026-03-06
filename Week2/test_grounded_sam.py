# codi adaptat de https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
from transformers import SamModel, SamProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from utils import grounded_segmentation
import cv2





image = Image.open("./000358.png").convert("RGB")

detector_id = "IDEA-Research/grounding-dino-tiny"
segmenter_id = "facebook/sam-vit-base"


image_array, detections = grounded_segmentation(
    image=image,
    labels=["a car."],
    threshold=0.3,
    polygon_refinement=True,
    detector_id=detector_id,
    segmenter_id=segmenter_id
)
     

# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].imshow(image)
# axes[0].set_title("Original")

# for i in range(3):
#     axes[i+1].imshow(image)
#     axes[i+1].imshow(mask[i].numpy(), alpha=0.5, cmap="Reds")
#     axes[i+1].set_title(f"Mask {i} | score: {scores[i]:.2f}")

# Convert PIL Image to OpenCV format
image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

# Iterate over detections and add bounding boxes and masks
for detection in detections:
    label = detection.label
    score = detection.score
    box = detection.box
    mask = detection.mask

    # Sample a random color for each detection
    color = np.random.randint(0, 256, size=3)

    # Draw bounding box
    cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
    cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    # If mask is available, apply it
    if mask is not None:
        # Convert mask to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

annotated_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

plt.imshow(annotated_image)
plt.axis('off')
plt.show()
