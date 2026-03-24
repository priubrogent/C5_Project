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

        self.samples = []
        for iid, fname in id2file.items():
            caps = img_captions.get(iid, [])
            self.samples.append((fname, caps))

        if split in ('train', 'val'):
            rng = random.Random(seed)
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            n_val = int(len(indices) * val_fraction)
            if split == 'val':
                keep = sorted(indices[:n_val])
            else:
                keep = sorted(indices[n_val:])
            self.samples = [self.samples[i] for i in keep]

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
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        img = self.img_proc(img)
        caption = random.choice(captions) if captions else ''
        encoded = self.tokenizer.encode(caption)
        return img, torch.tensor(encoded, dtype=torch.long), captions


def collate_fn(batch):
    imgs, encodings, all_captions = zip(*batch)
    return torch.stack(imgs), torch.stack(encodings), list(all_captions)


def explore_dataset(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    captions = [a['caption'] for a in data['annotations']]
    lengths = [len(c) for c in captions]
    words = [len(c.split()) for c in captions]
    print(f"Total images:   {len(data['images'])}")
    print(f"Total captions: {len(captions)}")
    print(f"Char length  -- mean: {sum(lengths)/len(lengths):.1f}, "
          f"max: {max(lengths)}, 99th pct: {sorted(lengths)[int(0.99*len(lengths))]}")
    print(f"Word length  -- mean: {sum(words)/len(words):.1f}, "
          f"max: {max(words)}, 99th pct: {sorted(words)[int(0.99*len(words))]}")
    print("\nSample captions:")
    for c in random.sample(captions, 5):
        print(f"  {c}")

def _clean_caption(
    caption: str,
    normalize_accents=True,
    keep_numbers=True
):
    caption = caption.lower()

    if normalize_accents:
        caption = unicodedata.normalize('NFKD', caption)
        caption = caption.encode('ascii', 'ignore').decode('ascii')

    caption = caption.replace('-', ' ')

    if keep_numbers:
        caption = re.sub(r"[^\w\s]", "", caption, flags=re.UNICODE)
    else:
        caption = ''.join(
            c for c in caption
            if unicodedata.category(c).startswith('L') or c.isspace()
        )

    caption = caption.replace('_', '')
    caption = re.sub(r"\s+", " ", caption).strip()

    return caption

def explore_dataset_word_frequencies(ann_file, clean=False):
    with open(ann_file) as f:
        data = json.load(f)
    
    captions = [a['caption'] for a in data['annotations']]
    if clean:
        captions = [_clean_caption(caption) for caption in captions]
    counter = Counter()
    for cap in captions:
        counter.update(cap.lower().split())
    
    word_frequency = sorted(counter.items(), key=lambda el: el[1], reverse=True)
    total = counter.total()
    for word, frequency in word_frequency:
        print(f"{word} -> {frequency} -> {frequency / total * 100:2f} %")
    

def explore_dataset_char_frequencies(ann_file, clean=False):
    with open(ann_file) as f:
        data = json.load(f)
    
    captions = [a['caption'] for a in data['annotations']]
    if clean:
        captions = [_clean_caption(caption) for caption in captions]
    words = [word for caption in captions for word in caption.lower().split()]
    chars = [char for word in words for char in word.split()]
    counter = Counter()
    for char in chars:
        counter.update(char)
    
    word_frequency = sorted(counter.items(), key=lambda el: el[1], reverse=True)
    total = counter.total()
    for word, frequency in word_frequency:
        print(f"{word} -> {frequency} -> {frequency / total * 100:2f} %")
    


if __name__ == '__main__':
    import sys
    ann = sys.argv[1] if len(sys.argv) > 1 else \
        '/home/priubrogent/01_MCV/C5/vizwiz_dataset/annotations/train.json'
    explore_dataset(ann)
    explore_dataset_word_frequencies(ann, clean=True)
    explore_dataset_char_frequencies(ann, clean=True)
