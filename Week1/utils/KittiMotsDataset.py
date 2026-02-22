import numpy as np
import os
import cv2
from torchvision.datasets import CocoDetection
from .utils import COCO_TO_DETR_ID, COCO_CLASSES

class KittiMotsDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, processor, sequence_ids, transform=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.transform = transform

        self.ids = [
            idx for idx in self.ids 
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]
        
    def __init__(self, img_folder, ann_file, processor, sequence_ids, transform=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.transform = transform

        self.ids = [
            idx for idx in self.ids 
            if (self.coco.loadImgs(idx)[0]['id'] // 100000) in sequence_ids
        ]

        # We only keep images and annotations belonging to the selected sequences
        val_img_ids_set = set(self.ids)
        
        self.coco.dataset['images'] = [
            img for img in self.coco.dataset['images'] if img['id'] in val_img_ids_set
        ]
        self.coco.dataset['annotations'] = [
            ann for ann in self.coco.dataset['annotations'] if ann['image_id'] in val_img_ids_set
        ]
        
        # REBUILD the index
        self.coco.createIndex()

    def __getitem__(self, idx):
        # 1. Load image and raw annotations
        img_id = self.ids[idx]
        img_metadata = self.coco.loadImgs(img_id)[0]
        # Use cv2 because Albumentations expects numpy arrays
        image = np.array(self.coco.loadImgs(img_id)[0]) # Placeholder logic, use your actual loader
        image = cv2.imread(os.path.join(self.root, img_metadata['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)
        
        # 2. Extract boxes and labels for Albumentations
        bboxes = []
        class_labels = []
        for ann in target:
            cat_id = ann['category_id']
            if cat_id in COCO_CLASSES and ann.get('iscrowd', 0) == 0:
                bboxes.append(ann['bbox'])
                class_labels.append(COCO_TO_DETR_ID[cat_id])

        # 3. Apply Albumentations
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        # 4. Format for the DETR Processor
        # The processor expects the annotations in COCO dict format
        new_target = []
        for box, label in zip(bboxes, class_labels):
            new_target.append({
                "category_id": label,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 0
            })

        encoding = self.processor(
            images=image, 
            annotations={'image_id': img_id, 'annotations': new_target}, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0), 
            "labels": encoding["labels"][0]
        }