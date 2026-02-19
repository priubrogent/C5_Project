from kitti_coco_mapping import KITTI_TO_COCO, COCO_CLASSES

def load_annotations(ann_path, frame_id):
    boxes, labels = [], []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != frame_id:
                continue
            obj_id = int(parts[1])
            class_id = obj_id // 1000
            if class_id not in KITTI_TO_COCO:
                continue
            h, w = int(parts[3]), int(parts[4])
            rle = {"counts": parts[5].encode(), "size": [h, w]}
            x, y, bw, bh = mask_util.toBbox(rle)
            boxes.append([x, y, x + bw, y + bh])
            labels.append(KITTI_TO_COCO[class_id])
    return boxes, labels