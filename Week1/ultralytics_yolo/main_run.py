import os

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import cv2
from ultralytics import YOLO
import argparse

from Week1.ultralytics_yolo.kitti_coco_mapping_yolo import KITTI_TO_COCO, COCO_CLASSES

#Opcions possibles depenent del size: yolov10n.pt, yolov10s.pt, yolov10m.pt, yolov10l.pt, yolov10x.pt
model = YOLO("yolov10s.pt")
model.to("cuda")

DATASET_PATH = "/data/113-2/users/gasbert/master/C5/KITTI-MOTS"
IMAGES_PATH = os.path.join(DATASET_PATH, "testing/image_02")
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "instances_txt")

allowed_classes = set(KITTI_TO_COCO.values())


def run_inference_on_dataset():
    sequence_folders = sorted(os.listdir(IMAGES_PATH))
    print(f"Found {len(sequence_folders)} sequences in the dataset.")

    for seq_folder in sequence_folders:
        seq_path = os.path.join(IMAGES_PATH, seq_folder)
        image_files = sorted(os.listdir(seq_path))
        print(f"Found {len(image_files)} images in sequence {seq_folder}.")

        for img_name in image_files:
            if not img_name.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(seq_path, img_name)
            image = cv2.imread(img_path)

            #YOLO inference
            results = model(image)

            result = results[0]
            
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            #Filtrem classes no desitjades
            keep_indices = [i for i, c in enumerate(classes) if c in allowed_classes]
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            classes = classes[keep_indices]

            print(f"\nImage: {img_name}")
            for box, score, cls in zip(boxes, scores, classes):
                print(f"Pred: Class={cls}, Conf={score:.3f}, Box={box}")

            #Visualitzem sol les prediccions que contenen les classes desitjades
            if len(keep_indices) > 0:
                filtered_result = result
                filtered_result.boxes = result.boxes[keep_indices]
                annotated_img = filtered_result.plot()
                cv2.imshow("YOLOv10 Filtered Predictions", annotated_img)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run YOLOv10 on KITTI-MOTS dataset")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["c"],
        help="Tasks of Week 1: c -> (YOLOv10 inference on KITTI-MOTS)",
    )
    args = parser.parse_args()

    if args.task == "c":
        run_inference_on_dataset()
