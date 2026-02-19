from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

KITTI_TO_COCO = {
    1: 2,
    2: 0,
}

COCO_CLASSES = set(KITTI_TO_COCO.values())
    