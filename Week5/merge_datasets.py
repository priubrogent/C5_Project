import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # Define all paths
    vizwiz_img_dir = Path("../datasets/vizwiz/train")
    # If you used the filtered dataset previously, change this to train_filtered.json
    vizwiz_json_path = Path("../datasets/vizwiz/annotations/train.json") 
    
    aug_img_dir = Path("./data_augmentation")
    aug_json_path = Path("./outputs/vizwiz_extended_captions.json")
    
    out_img_dir = Path("../datasets/vizwiz/train_augmented")
    out_json_path = Path("../datasets/vizwiz/annotations/train_augmented.json")
    
    # Create the output directories if they don't exist
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. MERGE THE JSON ANNOTATION FILES
    # ---------------------------------------------------------
    print(f"Loading original VizWiz JSON: {vizwiz_json_path}")
    with open(vizwiz_json_path, 'r') as f:
        vizwiz_data = json.load(f)
        
    print(f"Loading Augmented JSON: {aug_json_path}")
    with open(aug_json_path, 'r') as f:
        aug_data = json.load(f)

    # --- NEW CHECK: Deduplicate augmented data ---
    print("\n--- Checking for duplicates in Augmented Data ---")
    
    unique_aug_images = {}
    duplicate_img_count = 0
    for img in aug_data.get("images", []):
        if img["id"] in unique_aug_images:
            duplicate_img_count += 1
        else:
            unique_aug_images[img["id"]] = img
            
    unique_aug_annotations = {}
    duplicate_ann_count = 0
    for ann in aug_data.get("annotations", []):
        if ann["id"] in unique_aug_annotations:
            duplicate_ann_count += 1
        else:
            unique_aug_annotations[ann["id"]] = ann

    if duplicate_img_count > 0 or duplicate_ann_count > 0:
        print(f"[!] WARNING: Found and removed {duplicate_img_count} duplicate images and {duplicate_ann_count} duplicate annotations.")
    else:
        print("[✓] Passed! No duplicates found in the augmented JSON.")

    # Re-assign the cleanly filtered lists back
    clean_aug_images = list(unique_aug_images.values())
    clean_aug_annotations = list(unique_aug_annotations.values())
    # ---------------------------------------------
        
    # Combine the images and annotations lists
    merged_data = {
        # Keep original metadata if it exists
        "info": vizwiz_data.get("info", {}),
        "licenses": vizwiz_data.get("licenses", []),
        
        # Merge lists using the filtered augmented data
        "images": vizwiz_data.get("images", []) + clean_aug_images,
        "annotations": vizwiz_data.get("annotations", []) + clean_aug_annotations
    }
    
    print("\n--- Merge Statistics ---")
    print(f"Total images after merge:      {len(merged_data['images'])}")
    print(f"Total annotations after merge: {len(merged_data['annotations'])}")
    
    print(f"\nSaving merged JSON to: {out_json_path}")
    with open(out_json_path, 'w') as f:
        json.dump(merged_data, f, indent=4)
        
    # ---------------------------------------------------------
    # 2. COPY ALL IMAGES TO THE NEW DIRECTORY
    # ---------------------------------------------------------
    print(f"\nCopying original VizWiz images to {out_img_dir}...")
    vizwiz_images = [p for p in vizwiz_img_dir.iterdir() if p.is_file()]
    for img_path in tqdm(vizwiz_images, desc="Copying VizWiz"):
        shutil.copy2(img_path, out_img_dir / img_path.name)
            
    print(f"\nCopying augmented images to {out_img_dir}...")
    aug_images = [p for p in aug_img_dir.iterdir() if p.is_file()]
    for img_path in tqdm(aug_images, desc="Copying Augmented"):
        shutil.copy2(img_path, out_img_dir / img_path.name)
            
    print("\n" + "="*80)
    print("DATASET MERGE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()