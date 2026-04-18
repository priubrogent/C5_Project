import os
import json
import random

def analyze_quality_issues(data_root="../datasets/vizwiz", val_fraction=0.1, seed=42):
    train_ann = os.path.join(data_root, 'annotations', 'train.json')
    test_ann = os.path.join(data_root, 'annotations', 'val.json')
    
    # Helper function replicating the exact split logic from dataset.py
    def get_samples(ann_file, split):
        with open(ann_file) as f:
            data = json.load(f)
        
        id2file = {img['id']: img['file_name'] for img in data['images']}
        img_captions = {}
        for ann in data.get('annotations', []):
            iid = ann['image_id']
            img_captions.setdefault(iid, []).append(ann['caption'])
            
        samples = [(fname, img_captions.get(iid, [])) for iid, fname in id2file.items()]
        
        # Apply the exact same random split used during training
        if split in ('train', 'val'):
            rng = random.Random(seed)
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            n_val = int(len(indices) * val_fraction)
            if split == 'val':
                samples = [samples[i] for i in sorted(indices[:n_val])]
            else:
                samples = [samples[i] for i in sorted(indices[n_val:])]
                
        return samples

    # Load samples based on your train.py configuration
    print("Loading annotations...")
    splits = {
        'Train': get_samples(train_ann, 'train'),
        'Validation': get_samples(train_ann, 'val'),
        'Test': get_samples(test_ann, 'test')
    }
    
    target_phrase = "quality issues are too severe to recognize visual content"
    
    print("\n" + "=" * 65)
    print(f"{'Split':<15} | {'Issue Count':<15} | {'Total Images':<15} | {'Percentage':<10}")
    print("-" * 65)
    
    for name, samples in splits.items():
        issue_count = 0
        for fname, captions in samples:
            # Check if ANY caption contains the target phrase (case-insensitive)
            if any(target_phrase in cap.lower() for cap in captions):
                issue_count += 1
                
        total = len(samples)
        pct = (issue_count / total) * 100 if total > 0 else 0
        
        print(f"{name:<15} | {issue_count:<15} | {total:<15} | {pct:.2f}%")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    analyze_quality_issues()