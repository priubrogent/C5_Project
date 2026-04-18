import os
os.environ["DISABLE_TELEMETRY"] = "YES"
os.environ["DIFFUSERS_DISABLE_FLASH_ATTN"] = "1"

from pathlib import Path
import torch
import random
import time
import json
import wandb
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, DDPMScheduler

# Read objects list from txt file
objects_file = Path("/home/gasbert/Desktop/Master/C5/StableDiffusion/head_objects_gemini_filtered2.txt")
if objects_file.exists():
    with open(objects_file, 'r') as f:
        # Read the file, split by commas, and strip whitespace
        objects_list = [obj.strip() for obj in f.read().split(',')]
    print(f"✓ Loaded {len(objects_list)} objects from {objects_file}")
else:
    print(f"⚠ Warning: Objects file not found at {objects_file}")
    print("  Using default objects list.")
    # Fallback to default list if file doesn't exist
    objects_list = [
        "phone", "book", "cup", "keyboard", "mouse", "pen", "notebook", 
        "glasses", "watch", "wallet", "keys", "charger", "headphones",
        "lamp", "plant", "cup", "bottle", "plate", "fork", "spoon"
    ]

# Define prompt templates with placeholders
prompt_templates = [
    "A first-person photo taken indoors by a blind user, showing a [OBJECT_1], and a [OBJECT_2] in a messy environment. Some objects are partially occluded or hard to identify. The image has low lighting, noise, and an awkward angle.",
    "A blurry first-person photo taken indoors of a cluttered table taken by a visually impaired person, showing a [OBJECT_1] and a [OBJECT_2]. The objects are partially out of frame, with uneven lighting and slight motion blur. The scene looks unintentional and poorly centered, with some glare and shadows.",
    "A first-person POV image taken indoors of a [OBJECT_1] and a [OBJECT_2], taken very close and poorly framed, with parts of the objects cut off. The image is slightly out of focus, with harsh lighting and possible overexposure. The background is unclear and cluttered.",
    "A first-person view of a person holding or interacting with a [OBJECT_1]. The hand partially blocks the camera, and the image is slightly blurred. The framing is off-center, and the lighting is uneven, making details hard to see.",
    "A poorly captured first-person image showing a [OBJECT_1], taken by a visually impaired user. The image is dim, grainy, and slightly tilted, with objects not clearly visible and parts of the scene cut off."
]

# Define models to use
model_name = "StableDiffusion_XL"
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler_type = "ddim"
negative_prompt = None
cfg = 7.5
steps = 30


# Create base output directories
base_output_dir = Path("generated_images")
base_output_dir.mkdir(exist_ok=True)

final_images_dir = Path("final_images")
final_images_dir.mkdir(exist_ok=True)

# Dictionary to map image numbers to selected objects and prompts
image_objects_mapping = {}
inference_times_list = []

# Initialize wandb for the 1000 image generation run
run_name = "1000_img_generation"
wandb.init(project="stable-diffusion-exploration", name=run_name, reinit=True)

print(f"\n{'='*80}")
print(f"Loading model: {model_name} with scheduler: {scheduler_type}")
print(f"{'='*80}")

try:
    # Load pipeline based on model (once, outside the loop)
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

    # Choose scheduler
    if scheduler_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    
    # Log model and scheduler info to wandb
    wandb.log({
        "model": model_name,
        "scheduler": scheduler_type,
        "negative_prompt": negative_prompt,
        "num_steps": steps,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cfg": cfg
    })
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Generating 1000 images...\n")
    
    # Generate 1000 images with random prompts and objects
    for i in range(1000):
        try:
            # Select random prompt template
            template = random.choice(prompt_templates)
            
            # Select random objects (2 objects)
            selected_objects = random.sample(objects_list, min(2, len(objects_list)))
            
            # Fill in the placeholders
            prompt = template.replace("[OBJECT_1]", selected_objects[0])
            if "[OBJECT_2]" in prompt:
                prompt = prompt.replace("[OBJECT_2]", selected_objects[1] if len(selected_objects) > 1 else selected_objects[0])
            else:
                selected_objects = [selected_objects[0]]

            print("Selected Objects: ", selected_objects)
            print("Prompt: ", prompt)
            
            # Record start time
            inference_start = time.time()
            
            # Generate image
            with torch.inference_mode():
                image = pipe(prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=steps,
                            guidance_scale=cfg).images[0]
            
            # Record end time and calculate inference time
            inference_end = time.time()
            inference_time = inference_end - inference_start
            inference_times_list.append(inference_time)
            
            # Save image with iteration number as filename
            filename = f"{i}.png"
            output_path = final_images_dir / filename
            image.save(output_path)
            
            # Store mapping of image number to objects and prompt
            image_objects_mapping[str(i)] = {
                "objects": selected_objects,
                "prompt": prompt
            }
            
            # Print progress
            if (i + 1) % 100 == 0:
                print(f"  ✓ Generated {i + 1}/1000 images (Latest inference time: {inference_time:.2f}s)")
            
            # Log to wandb every 10 images
            if (i + 1) % 10 == 0:
                wandb.log({
                    "iteration": i,
                    "objects": str(selected_objects),
                    "prompt": prompt,
                    "inference_time": inference_time,
                    "image": wandb.Image(str(output_path), caption=f"Iteration {i}: {', '.join(selected_objects)}")
                })
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Error at iteration {i}: {e}")
    
    # Free memory after all images are generated
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Finish wandb run
    wandb.finish()
    
    print(f"\n✓ All 1000 images generated successfully!")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    wandb.finish()

# Print statistics
print("\n" + "="*80)
print("Generation Statistics")
print("="*80)
if inference_times_list:
    avg_time = sum(inference_times_list) / len(inference_times_list)
    min_time = min(inference_times_list)
    max_time = max(inference_times_list)
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Minimum inference time: {min_time:.2f}s")
    print(f"Maximum inference time: {max_time:.2f}s")
    print(f"Total time: {sum(inference_times_list):.2f}s")

# Save image-objects mapping to JSON file
json_output_path = final_images_dir / "image_metadata.json"
with open(json_output_path, 'w') as f:
    json.dump(image_objects_mapping, f, indent=2)
print(f"\n✓ Image metadata saved to: {json_output_path}")
