from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

KITTI_TO_COCO = {
    1: 3,
    2: 1,
}

COCO_CLASSES = set(KITTI_TO_COCO.values())
    

if __name__ == "__main__":
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    COCO_LABELS= weights.meta["categories"]
    for i, label in enumerate(COCO_LABELS):
        print(i, label)