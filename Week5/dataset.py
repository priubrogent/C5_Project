from collections import Counter
import json
import os
import random
import re
import unicodedata
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class VizWizDataset(Dataset):

    def __init__(self, img_dir, ann_file, tokenizer, split='train',
                 val_fraction=0.1, seed=42):
        self.img_dir = img_dir
        self.tokenizer = tokenizer

        with open(ann_file) as f:
            data = json.load(f)

        id2file = {img['id']: img['file_name'] for img in data['images']}

        img_captions = {}
        for ann in data.get('annotations', []):
            iid = ann['image_id']
            img_captions.setdefault(iid, []).append(ann['caption'])

        original_samples = []
        augmented_samples = []
        
        # Separate original VizWiz images from our generated ones
        # We know our generated images start with IDs >= 1,000,000
        for iid, fname in id2file.items():
            caps = img_captions.get(iid, [])
            if iid >= 1000000:
                augmented_samples.append((fname, caps))
            else:
                original_samples.append((fname, caps))

        if split in ('train', 'val'):
            rng = random.Random(seed)
            # Only shuffle and split the original VizWiz images
            indices = list(range(len(original_samples)))
            rng.shuffle(indices)
            n_val = int(len(indices) * val_fraction)
            
            if split == 'val':
                # Validation gets ONLY a pure subset of original images
                keep = sorted(indices[:n_val])
                self.samples = [original_samples[i] for i in keep]
            else:
                # Train gets the rest of the original images PLUS all augmented images
                keep = sorted(indices[n_val:])
                self.samples = [original_samples[i] for i in keep] + augmented_samples
        else:
            # For 'test' split, just load whatever is in the file
            self.samples = original_samples + augmented_samples

        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, captions = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.img_proc(img)

        # Ensure we always have at least one valid caption to avoid errors
        if not captions:
            captions = [""]
            
        caption = random.choice(captions)
        encoded = self.tokenizer.encode(caption)
        
        return img_tensor, torch.tensor(encoded, dtype=torch.long), captions

def collate_fn(batch):
    imgs, caps, all_caps = zip(*batch)
    imgs = torch.stack(imgs)
    
    max_len = max(len(c) for c in caps)
    padded_caps = []
    
    # Needs access to the tokenizer's PAD token ID, ideally passed or accessible.
    # We'll use 2 as a safe default based on CharTokenizer, but it usually comes from the dataset
    pad_idx = 2 
    for c in caps:
        pad_len = max_len - len(c)
        padded = torch.cat([c, torch.full((pad_len,), pad_idx, dtype=torch.long)])
        padded_caps.append(padded)
        
    return imgs, torch.stack(padded_caps), all_caps