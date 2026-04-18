import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    img_dir = "./final_images"  # Update this to your images folder
    json_path = "image_metadata.json"
    out_path = "./outputs/vizwiz_extended_captions.json"
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Load metadata
    print(f"Loading metadata from {json_path}...")
    with open(json_path, 'r') as f:
        metadata = json.load(f)
        
    # Initialize Qwen 9B Model and Processor
    model_id = "Qwen/Qwen3.5-9B-Base"
    print(f"Loading {model_id} on {device}...")
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # VizWiz / COCO format structure
    vizwiz_output = {
        "info": {"description": "Extended dataset generated with Qwen3.5-9B-Base (1st sentence only)"},
        "images": [],
        "annotations": []
    }

    batch_size = 4  # Adjust based on your available VRAM
    keys = list(metadata.keys())
    
    # Counters for VizWiz IDs
    image_id_counter = 1000000  # Starting high to avoid clashing with original dataset IDs
    annotation_id_counter = 1000000

    print("\n" + "="*80)
    print("STARTING BATCHED INFERENCE (VIZWIZ FORMAT)")
    print("="*80)

    with torch.no_grad():
        for i in tqdm(range(0, len(keys), batch_size), desc="Captioning Images"):
            batch_keys = keys[i:i+batch_size]
            images = []
            texts = []
            valid_batch_info = []
            
            for k in batch_keys:
                img_name = k if "." in k else f"{k}.png"
                img_path = os.path.join(img_dir, img_name)
                
                if not os.path.exists(img_path):
                    continue
                    
                images.append(Image.open(img_path).convert("RGB"))
                
                # Extract objects to guide the model
                objects_list = metadata[k].get("objects", [])
                objects_str = ", ".join(objects_list)
                
                # Modified prompt forcing objects into the first sentence
                text_prompt = (
                    "<|vision_start|><|image_pad|><|vision_end|>\n"
                    "Domain: Everyday casual photography.\n"
                    f"Requirement: You MUST include these specific objects in your very first sentence: {objects_str}.\n"
                    "Style: Simple, brief visual description. One single sentence. It is important not to output technical specifications or numbers.\n"
                    "Image caption: "
                )
                texts.append(text_prompt)
                
                # Store info needed for the JSON structure
                valid_batch_info.append({
                    "file_name": img_name,
                    "image_id": image_id_counter
                })
                image_id_counter += 1
                
            if not images:
                continue
                
            # Run inference on the batch
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            # Max new tokens can be kept reasonably short since we only want the first sentence
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            
            # Format, strict truncation, and append to VizWiz structure
            for info, pred in zip(valid_batch_info, preds):
                # Cut at the first period and append the period back
                pred_clean = pred.strip().split('.')[0] + '.'
                
                # Add to images list
                vizwiz_output["images"].append({
                    "id": info["image_id"],
                    "file_name": info["file_name"]
                })
                
                # Add to annotations list
                vizwiz_output["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": info["image_id"],
                    "caption": pred_clean
                })
                annotation_id_counter += 1

    # Save output to JSON
    with open(out_path, "w") as f:
        json.dump(vizwiz_output, f, indent=4)
        
    print(f"\nCompleted! VizWiz-formatted dataset saved to {out_path}")

if __name__ == "__main__":
    main()