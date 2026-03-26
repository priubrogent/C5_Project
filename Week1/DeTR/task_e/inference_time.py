import torch
import time
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from peft import PeftModel
import os
import torch.nn as nn  

BACKBONE_ABLATION = {
    1: 3, # 3 in total
    2: 4, # 4 in total
    3: 6, # 6 in total
    4: 1  # 3 in total
}

# --- Configuration (Same as your qualitative script) ---
LORA_ADAPTER_DIR = "DeTR/Results_DETR/task_e/ablation_3_4_6_1_lora_adapter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prune_resnet_backbone(model, pruning_dict):
    """
    Updated for Hugging Face DetrForObjectDetection structure.
    pruning_dict: {stage_number: blocks_to_keep}
    """
    # 1. Reach the internal DETR model
    # If using PEFT, we go through get_base_model()
    curr_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    
    # 2. Correct path to the ResNet object in HF Transformers
    # model.model.backbone -> DetrConvEncoder
    # model.model.backbone.backbone -> DetrResnetBackbone
    # model.model.backbone.backbone.model -> The actual ResNet with layer1, layer2, etc.
    try:
        backbone = curr_model.model.backbone.backbone.model
    except AttributeError:
        # Fallback for some versions of the library
        backbone = curr_model.model.backbone.model
        
    print(f"Successfully reached backbone: {type(backbone).__name__}")

    for stage_num, keep_count in pruning_dict.items():
        stage_name = f"layer{stage_num}"
        if not hasattr(backbone, stage_name):
            print(f"Warning: Stage {stage_num} ({stage_name}) not found. Skipping.")
            continue
            
        stage = getattr(backbone, stage_name)
        total_blocks = len(stage)
        
        # Guard: Ensure we keep at least the downsampling block (index 0)
        if keep_count < 1:
            keep_count = 1
            
        if keep_count >= total_blocks:
            continue

        # Replace unwanted blocks with Identity
        for i in range(keep_count, total_blocks):
            stage[i] = nn.Identity()
            
        print(f"Pruned Stage {stage_num}: Reduced from {total_blocks} to {keep_count} active blocks.")

def measure_inference_speed():
    # 1. Load Model and Config
    print(f"Loading model on {device}...")
    config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    config.num_labels = 2
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    base_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", 
        config=config, 
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR).to(device)
    prune_resnet_backbone(model, BACKBONE_ABLATION)
    model.eval()

    # 2. Create Dummy Input (Standard DETR size 1242x375 for KITTI)
    # Using a typical batch size of 1 for real-time inference testing
    dummy_input = torch.randn(1, 3, 1242, 375).to(device)
    
    # 3. GPU Warm-up
    # Essential for accurate CUDA timing
    print("Warming up the GPU...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    # 4. Benchmark Loop
    num_iterations = 100
    latencies = []
    
    print(f"Running {num_iterations} iterations...")
    with torch.no_grad():
        for _ in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize() # Wait for GPU to be ready
            
            start_time = time.time()
            
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize() # Wait for GPU to finish
            
            end_time = time.time()
            latencies.append(end_time - start_time)

    # 5. Results Analysis
    avg_latency_ms = np.mean(latencies) * 1000
    std_latency_ms = np.std(latencies) * 1000
    fps = 1 / np.mean(latencies)

    print("\n" + "="*30)
    print(f"INFERENCE RESULTS ({device})")
    print("="*30)
    print(f"Average Latency: {avg_latency_ms:.2f} ms")
    print(f"Std Deviation:   {std_latency_ms:.2f} ms")
    print(f"Throughput:      {fps:.2f} FPS")
    print("="*30)

if __name__ == "__main__":
    measure_inference_speed()