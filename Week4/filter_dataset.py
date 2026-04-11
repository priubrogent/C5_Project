import json
import os

def filter_dataset(input_json, output_json):
    target_phrase = "quality issues are too severe to recognize visual content"
    
    print(f"Processing {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
        
    original_ann_count = len(data.get('annotations', []))
    original_img_count = len(data.get('images', []))
    
    # 1. Filter annotations
    filtered_annotations = []
    valid_image_ids = set()
    
    for ann in data.get('annotations', []):
        # Case-insensitive check to catch variations
        if target_phrase not in ann['caption'].lower():
            filtered_annotations.append(ann)
            valid_image_ids.add(ann['image_id'])
            
    # 2. Filter images (keep only images that still have at least 1 valid caption)
    filtered_images = [img for img in data.get('images', []) if img['id'] in valid_image_ids]
    
    # 3. Reconstruct JSON (preserving other COCO metadata like info/licenses)
    new_data = {}
    for key in data.keys():
        if key == 'annotations':
            new_data[key] = filtered_annotations
        elif key == 'images':
            new_data[key] = filtered_images
        else:
            new_data[key] = data[key]
            
    # 4. Save to new file
    with open(output_json, 'w') as f:
        json.dump(new_data, f)
        
    print(f"  Annotations: {original_ann_count} -> {len(filtered_annotations)}")
    print(f"  Images:      {original_img_count} -> {len(filtered_images)}")
    print(f"  Saved to:    {output_json}\n")

def main():
    data_root = "../datasets/vizwiz"
    ann_dir = os.path.join(data_root, 'annotations')
    
    # Define paths
    train_input = os.path.join(ann_dir, 'train.json')
    val_input = os.path.join(ann_dir, 'val.json')
    
    train_output = os.path.join(ann_dir, 'train_filtered.json')
    val_output = os.path.join(ann_dir, 'val_filtered.json')
    
    # Process files
    if os.path.exists(train_input):
        filter_dataset(train_input, train_output)
    else:
        print(f"[!] File not found: {train_input}")
        
    if os.path.exists(val_input):
        filter_dataset(val_input, val_output)
    else:
        print(f"[!] File not found: {val_input}")

if __name__ == "__main__":
    main()