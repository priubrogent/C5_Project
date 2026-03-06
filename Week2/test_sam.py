from transformers import SamModel, SamProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
plt.ion()
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
model.eval()

image = Image.open("./000358.png").convert("RGB")

inputs = processor(images=image, input_points=[[[789,193]]], return_tensors="pt").to(device)
print(inputs)
with torch.no_grad():
    outputs = model(**inputs)


masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

mask = masks[0][0]
scores = outputs.iou_scores[0,0]

# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# axes[0].imshow(image)
# axes[0].set_title("Original")

# for i in range(3):
#     axes[i+1].imshow(image)
#     axes[i+1].imshow(mask[i].numpy(), alpha=0.5, cmap="Reds")
#     axes[i+1].set_title(f"Mask {i} | score: {scores[i]:.2f}")


fig = plt.figure
plt.imshow(image)
plt.imshow(mask[0].numpy(), alpha=0.5, cmap='Reds')