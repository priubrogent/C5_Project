import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

COCO_LABELS = weights.meta["categories"]

# dataset_path = "/hhome/mcv/datasets/C5/KITTI-MOTS/testing/image_02"
dataset_path = "../KITTI-MOTS/training/image_02/0011"
output_dir = "results_rcnn"
os.makedirs(output_dir, exist_ok=True)

images = sorted(os.listdir(dataset_path))[:10]

for img_name in images:
    image = Image.open(os.path.join(dataset_path, img_name)).convert("RGB")
    img_tensor = F.to_tensor(image)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    fig, ax = plt.subplots(1, figsize=(12, 6))
    ax.imshow(image)

    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < 0.5:
            continue
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{COCO_LABELS[label]} {score:.2f}", color="red", fontsize=12)

    plt.savefig(os.path.join(output_dir, img_name))
    plt.close()