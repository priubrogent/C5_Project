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

schedulers = [
    "ddim", 
    #"ddpm"
]

negative_prompts = {
    0: None, #No negative prompts
    #1: "disfigured objects, bad anatomy",
    #2:  "deformed objects, disfigured hands",
}

cfg_scales = [
    #2.5,
    #5.0, 
    7.5,
    #10,
    #12.5
]

steps_list = [
#    10, 
#    20, 
    30, 
#    40,
#    50
]


# Create base output directory
base_output_dir = Path("generated_images")
base_output_dir.mkdir(exist_ok=True)

# Dictionary to track inference times per model-scheduler pair
inference_times = {}
for model_name in models.keys():
    inference_times[model_name] = {}
    for scheduler_type in schedulers:
        inference_times[model_name][scheduler_type] = []

# Dictionary to map image names to selected objects
image_objects_mapping = {}

# Iterate through each model and scheduler pair
for model_name, model_id in models.items():
    for scheduler_type in schedulers:
        for neg_prompt_id, negative_prompt in negative_prompts.items():
            for cfg in cfg_scales:
                for steps in steps_list:
                    # Initialize wandb run for this model-scheduler pair
                    run_name = f"{model_name}_{scheduler_type}_negprompt" + str(neg_prompt_id) + "cfg" + str(cfg) + "steps" + str(steps)
                    wandb.init(project="stable-diffusion-exploration", name=run_name, reinit=True)
                    
                    print(f"\n{'='*80}")
                    print(f"Loading model: {model_name} with scheduler: {scheduler_type}")
                    print(f"{'='*80}")
                    
                    # Create folder with all configuration options in the name
                    config_folder_name = f"{model_name}_{scheduler_type}_negprompt{neg_prompt_id}_cfg{cfg}_steps{steps}"
                    model_output_dir = base_output_dir / config_folder_name
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
                            "neg_prompt_id": neg_prompt_id,
                            "negative_prompt": negative_prompt,
                            "num_steps": steps,
                            "device": "cuda" if torch.cuda.is_available() else "cpu",
                            "cfg": cfg
                        })
                        
                        # Generate 50 images with random prompts and objects
                        for iteration in range(50):
                            # Select random prompt template
                            template = random.choice(prompt_templates)
                            
                            # Select random objects
                            selected_objects = random.sample(objects_list, min(2, len(objects_list)))
                                            
                            # Fill in the placeholders
                            prompt = template.replace("[OBJECT_1]", selected_objects[0])
                            if "[OBJECT_2]" in prompt:
                                prompt = prompt.replace("[OBJECT_2]", selected_objects[1] if len(selected_objects) > 1 else selected_objects[0])
                            else:
                                selected_objects = [selected_objects[0]]
                            
                            # Filename is just the iteration number
                            filename = f"{iteration}.png"
                            
                            # Store mapping of selected objects
                            config_key = f"{config_folder_name}/{filename}"
                            if config_key not in image_objects_mapping:
                                image_objects_mapping[config_key] = {
                                    "objects": selected_objects,
                                    "prompt": prompt
                                }
                            
                            if (iteration + 1) % 10 == 0:
                                print(f"\n  Generating iteration {iteration + 1}/50: Objects: {', '.join(selected_objects)}")
                            
                            try:
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
                                inference_times[model_name][scheduler_type].append(inference_time)
                                
                                # Save image
                                output_path = model_output_dir / filename
                                image.save(output_path)
                                if (iteration + 1) % 10 == 0:
                                    print(f"    ✓ Saved iteration {iteration + 1}: {output_path}")
                                    print(f"      Inference time: {inference_time:.2f}s")
                                
                                # Log to wandb every 5 iterations
                                if (iteration + 1) % 5 == 0:
                                    wandb.log({
                                        "iteration": iteration,
                                        "objects": str(selected_objects),
                                        "prompt": prompt,
                                        "inference_time": inference_time,
                                        "image": wandb.Image(str(output_path), caption=f"Iteration {iteration}: {', '.join(selected_objects)}")
                                    })
                                
                            except Exception as e:
                                print(f"    ✗ Error at iteration {iteration}: {e}")

                            torch.cuda.empty_cache()
                        
                        # Free memory after all iterations for this configuration
                        del pipe
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Finish wandb run
                        wandb.finish()
                                
                    except Exception as e:
                        print(f"✗ Error loading model {model_name} with scheduler {scheduler_type}: {e}")
                        wandb.finish()


# Print inference times table
print("\n" + "="*80)
print("Inference Times Summary (seconds)")
print("="*80)

# Create header
header = f"{'Model':<30} {'DDIM':<15} {'DDPM':<15}"
print(header)
print("-" * 60)

# Print rows for each model
for model_name in models.keys():
    ddim_times = inference_times[model_name].get("ddim", [])
    ddpm_times = inference_times[model_name].get("ddpm", [])
                            
    ddim_avg = sum(ddim_times) / len(ddim_times) if ddim_times else 0
    ddpm_avg = sum(ddpm_times) / len(ddpm_times) if ddpm_times else 0
                            
    print(f"{model_name:<30} {ddim_avg:<15.2f} {ddpm_avg:<15.2f}")


# Save image-objects mapping to JSON file
json_output_path = base_output_dir / "image_objects_mapping.json"
with open(json_output_path, 'w') as f:
    json.dump(image_objects_mapping, f, indent=2)
print(f"\n✓ Image-objects mapping saved to: {json_output_path}")

print("\n✓ Image generation complete!")
print(f"Images saved in: {base_output_dir.absolute()}")
print(f"✓ Check wandb for detailed logs at: https://wandb.ai")
