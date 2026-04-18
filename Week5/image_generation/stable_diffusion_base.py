import os
os.environ["DISABLE_TELEMETRY"] = "YES"
os.environ["DIFFUSERS_DISABLE_FLASH_ATTN"] = "1"

from pathlib import Path
import torch
import random
import time
import json
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

# List of objects to randomly insert
objects_list = [
    "phone", "book", "cup", "keyboard", "mouse", "pen", "notebook", 
    "glasses", "watch", "wallet", "keys", "charger", "headphones",
    "lamp", "plant", "cup", "bottle", "plate", "fork", "spoon"
]

# Define prompt templates with placeholders
prompt_templates = [
    "A first-person photo taken indoors by a blind user, showing a [OBJECT_1], and a [OBJECT_2] in a messy environment. Some objects are partially occluded or hard to identify. The image has low lighting, noise, and an awkward angle.",
    "A blurry first-person photo of a cluttered table taken by a visually impaired person, showing a [OBJECT_1] and a [OBJECT_2]. The objects are partially out of frame, with uneven lighting and slight motion blur. The scene looks unintentional and poorly centered, with some glare and shadows.",
    "A first-person POV image of [OBJECT_1] and [OBJECT_2], taken very close and poorly framed, with parts of the objects cut off. The image is slightly out of focus, with harsh lighting and possible overexposure. The background is unclear and cluttered.",
    "A first-person view of a person holding or interacting with a [OBJECT_1]. The hand partially blocks the camera, and the image is slightly blurred. The framing is off-center, and the lighting is uneven, making details hard to see.",
    "A poorly captured first-person image showing a [OBJECT_1], taken by a visually impaired user. The image is dim, grainy, and slightly tilted, with objects not clearly visible and parts of the scene cut off."
]

# Define models to use
models = {
    "StableDiffusion_2.1": "sd2-community/stable-diffusion-2-1",
    "StableDiffusion_XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "StableDiffusion_Turbo": "stabilityai/sd-turbo",
}

# Create base output directory
base_output_dir = Path("generated_images")
base_output_dir.mkdir(exist_ok=True)

# Dictionary to track inference times per model
inference_times = {model_name: [] for model_name in models.keys()}

# Dictionary to map image names to selected objects
image_objects_mapping = {}

# Iterate through each prompt template
for prompt_idx, template in enumerate(prompt_templates):
    # Select random objects
    selected_objects = random.sample(objects_list, min(2, len(objects_list)))
    
    # Fill in the placeholders
    prompt = template.replace("[OBJECT_1]", selected_objects[0])
    if "[OBJECT_2]" in prompt:
        prompt = prompt.replace("[OBJECT_2]", selected_objects[1] if len(selected_objects) > 1 else selected_objects[0])
    else:
        selected_objects = [selected_objects[0]]
    
    # Create filename from prompt index and objects
    objects_str = "_".join(selected_objects)
    filename = f"prompt_{prompt_idx}_{objects_str}.png"
    
    print(f"\n{'='*80}")
    print(f"Processing Prompt {prompt_idx + 1}/5: Objects: {', '.join(selected_objects)}")
    print(f"{'='*80}")
    
    # Store mapping of selected objects for all models that will be generated
    image_objects_mapping[filename] = selected_objects

    # Generate images with each model
    for model_name, model_id in models.items():
        print(f"\nGenerating image with {model_name}...")
        
        # Create model-specific folder
        model_output_dir = base_output_dir / model_name
        model_output_dir.mkdir(exist_ok=True)
        
        try:
            # Load pipeline based on model
            if "XL" in model_name:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if torch.cuda.is_available() else None
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if torch.cuda.is_available() else None
                )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                pipe.enable_attention_slicing()
            
            # Record start time
            inference_start = time.time()
            
            # Generate image
            with torch.inference_mode():
                if model_name == "StableDiffusion_Turbo":
                    # Turbo is faster with fewer steps
                    image = pipe(prompt, num_inference_steps=1).images[0]
                else:
                    # Standard models
                    image = pipe(prompt, num_inference_steps=30).images[0]
            
            # Record end time and calculate inference time
            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_times[model_name].append(inference_time)
            
            # Save image
            output_path = model_output_dir / filename
            image.save(output_path)
            print(f"✓ Saved: {output_path}")
            print(f"  Inference time: {inference_time:.2f}s")
            
            # Free memory
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"✗ Error generating image with {model_name}: {e}")

# Print mean inference times for each model
print("\n" + "="*80)
print("INFERENCE TIME STATISTICS")
print("="*80)
for model_name, times in inference_times.items():
    if times:
        mean_time = sum(times) / len(times)
        print(f"{model_name}: {mean_time:.2f}s (average from {len(times)} images)")
    else:
        print(f"{model_name}: No images generated")

# Save image-objects mapping to JSON file
json_output_path = base_output_dir / "image_objects_mapping.json"
with open(json_output_path, 'w') as f:
    json.dump(image_objects_mapping, f, indent=2)
print(f"\n✓ Image-objects mapping saved to: {json_output_path}")

print("\n✓ Image generation complete!")
print(f"Images saved in: {base_output_dir.absolute()}")
