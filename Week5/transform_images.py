import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def resize_to_vizwiz_quality(image_np, max_dim=640):
    """Reduces dimensions to match typical phone camera resolutions found in VizWiz."""
    h, w = image_np.shape[:2]
    if max(h, w) <= max_dim:
        return image_np
    
    # Calculate aspect ratio
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_AREA for downsampling to avoid aliasing artifacts
    return cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

def apply_motion_blur(image_np, kernel_size=15, angle=0):
    """Applies directional motion blur to simulate shaky hands."""
    kernel = np.zeros((kernel_size, kernel_size))
    center = int((kernel_size - 1) / 2)
    kernel[center, :] = np.ones(kernel_size)
    
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    
    return cv2.filter2D(image_np, -1, kernel)

def save_with_jpeg_compression(image_np, output_path, quality=30):
    """Saves the image with heavy JPEG compression artifacts."""
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    pil_img.save(str(output_path), "JPEG", quality=quality)
    
def apply_overexposure(image_np, factor=1.8, offset=50):
    """Simulates very bright light by clipping pixel values to solid white."""
    bright_img = image_np.astype(np.float32) * factor + offset
    return np.clip(bright_img, 0, 255).astype(np.uint8)

def main():
    img_dir = Path("./final_images") # Folder containing original high-quality images
    output_base = Path("./data_augmentation") # New target folder for all resized/augmented images
    json_path = Path("./outputs/vizwiz_extended_captions.json")
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the existing extended VizWiz JSON
    print(f"Loading metadata from {json_path}...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Create lookup dictionaries to easily find the original caption
    image_id_to_caption = {ann['image_id']: ann['caption'] for ann in coco_data.get('annotations', [])}
    filename_to_img_id = {img['file_name']: img['id'] for img in coco_data.get('images', [])}
    
    # Find the starting IDs for the new entries to avoid collisions
    next_img_id = max([img['id'] for img in coco_data.get('images', [])] + [999999]) + 1
    next_ann_id = max([ann['id'] for ann in coco_data.get('annotations', [])] + [999999]) + 1
    
    # Gather all original images
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [
        p for p in img_dir.iterdir() 
        if p.suffix.lower() in valid_extensions 
        and not p.name.startswith("mod_") 
        and not p.name.startswith("sev_")
    ]
    
    print(f"Found {len(image_paths)} original images. Starting downscaling and augmentation...")
    rng = np.random.default_rng(42)
    
    for img_path in tqdm(image_paths, desc="Processing Images"):
        filename = img_path.name
        
        # Skip if the image isn't in our JSON metadata
        if filename not in filename_to_img_id:
            continue
            
        orig_img_id = filename_to_img_id[filename]
        orig_caption = image_id_to_caption.get(orig_img_id, "")
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # ---------------------------------------------------------
        # 0. REDUCE DIMENSIONS (VizWiz-like Quality)
        # ---------------------------------------------------------
        # Downscale the image to a maximum of 500px on the longest side
        standard_img = resize_to_vizwiz_quality(img, max_dim=500)
        
        # Save the downscaled original image to the new folder
        standard_out_path = output_base / filename
        cv2.imwrite(str(standard_out_path), standard_img)
            
        # ---------------------------------------------------------
        # 1. MODERATE DEGRADATION (Keeps the original caption)
        # ---------------------------------------------------------
        mod_filename = f"mod_{filename}"
        
        # Apply augmentations to the already downscaled image
        mod_img = apply_motion_blur(standard_img, kernel_size=12, angle=np.random.randint(0, 180))
        mod_img = apply_overexposure(mod_img, factor=(0.2*rng.random()+1.0), offset=20*rng.random())
        
        save_with_jpeg_compression(mod_img, output_base / mod_filename, quality=60)
        
        """# Append to JSON
        coco_data["images"].append({
            "id": next_img_id,
            "file_name": mod_filename
        })
        coco_data["annotations"].append({
            "id": next_ann_id,
            "image_id": next_img_id,
            "caption": orig_caption
        })"""
        
        next_img_id += 1
        next_ann_id += 1
        
        # ---------------------------------------------------------
        # 2. SEVERE DEGRADATION (Uses the VizWiz boilerplate caption)
        # ---------------------------------------------------------
        sev_filename = f"sev_{filename}"
        
        # Apply severe augmentations to the downscaled image
        if np.random.rand() > 0.5:
            sev_img = apply_overexposure(standard_img, factor=2.75, offset=100)
        else:
            sev_img = apply_motion_blur(standard_img, kernel_size=75, angle=np.random.randint(0, 180))
            
        save_with_jpeg_compression(sev_img, output_base / sev_filename, quality=40)
        
        """# Append to JSON
        coco_data["images"].append({
            "id": next_img_id,
            "file_name": sev_filename
        })
        coco_data["annotations"].append({
            "id": next_ann_id,
            "image_id": next_img_id,
            "caption": "Quality issues are too severe to recognize visual content."
        })"""
        
        next_img_id += 1
        next_ann_id += 1

    # Save the extended JSON back to disk
    print(f"\nSaving updated annotations to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"Processing complete! All resized and augmented images are stored in: {output_base}")

if __name__ == "__main__":
    main()