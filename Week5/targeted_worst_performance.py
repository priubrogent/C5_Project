import os
import shutil
import json
import torch
import evaluate
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from dataset import VizWizDataset, collate_fn
from tokenizer import build_tokenizer
from models import CaptioningModel

# The exact list of images you want to evaluate
TARGET_IMAGES = [
    "VizWiz_val_00000111.jpg", "VizWiz_val_00000870.jpg", "VizWiz_val_00000458.jpg",
    "VizWiz_val_00000703.jpg", "VizWiz_val_00000514.jpg", "VizWiz_val_00000639.jpg",
    "VizWiz_val_00000863.jpg", "VizWiz_val_00000835.jpg", "VizWiz_val_00000613.jpg",
    "VizWiz_val_00000085.jpg", "VizWiz_val_00000657.jpg", "VizWiz_val_00000673.jpg",
    "VizWiz_val_00000826.jpg", "VizWiz_val_00000196.jpg", "VizWiz_val_00000138.jpg",
    "VizWiz_val_00000596.jpg", "VizWiz_val_00000808.jpg", "VizWiz_val_00000324.jpg",
    "VizWiz_val_00000535.jpg", "VizWiz_val_00000853.jpg", "VizWiz_val_00000010.jpg",
    "VizWiz_val_00000753.jpg", "VizWiz_val_00000756.jpg", "VizWiz_val_00000008.jpg",
    "VizWiz_val_00000389.jpg", "VizWiz_val_00000510.jpg", "VizWiz_val_00000028.jpg",
    "VizWiz_val_00000297.jpg", "VizWiz_val_00000371.jpg", "VizWiz_val_00000638.jpg",
    "VizWiz_val_00000393.jpg", "VizWiz_val_00000578.jpg", "VizWiz_val_00000181.jpg",
    "VizWiz_val_00000270.jpg", "VizWiz_val_00000141.jpg", "VizWiz_val_00000574.jpg",
    "VizWiz_val_00000867.jpg", "VizWiz_val_00000820.jpg", "VizWiz_val_00000831.jpg",
    "VizWiz_val_00000590.jpg", "VizWiz_val_00000368.jpg", "VizWiz_val_00000352.jpg",
    "VizWiz_val_00000107.jpg", "VizWiz_val_00000513.jpg", "VizWiz_val_00000390.jpg",
    "VizWiz_val_00000771.jpg", "VizWiz_val_00000770.jpg", "VizWiz_val_00000789.jpg",
    "VizWiz_val_00000901.jpg", "VizWiz_val_00000631.jpg"
]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "../datasets/vizwiz"
    out_dir = "./outputs/targeted_predictions"
    batch_size = 32  
    
    # 1. Setup Test Dataset
    val_img_dir = os.path.join(data_root, 'val')
    val_ann = os.path.join(data_root, 'annotations', 'val.json') 
    cache_dir = os.path.join(data_root, 'tokenizer_cache')
    
    print("Loading test dataset...")
    tokenizer = build_tokenizer('subword', val_ann, cache_dir)
    ds_test = VizWizDataset(val_img_dir, val_ann, tokenizer, split='test', seed=42)
    
    # --- FILTER DATASET TO ONLY INCLUDE TARGET IMAGES ---
    filtered_samples = [s for s in ds_test.samples if s[0] in TARGET_IMAGES]
    ds_test.samples = filtered_samples
    print(f"Dataset successfully filtered to {len(ds_test.samples)} target images.")
    
    # Create DataLoader (shuffle MUST be False to map filenames correctly)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # Load METEOR metric
    meteor_metric = evaluate.load('meteor')

    # 2. Load Model (CLIP + Qwen 0.8B)
    print("Loading Qwen 0.8B Model...")
    model = CaptioningModel(
        encoder_name='clip', 
        decoder_type='qwen',
        decoder_model_name='Qwen/Qwen3.5-0.8B-Base',
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768
    ).to(device)
    
    # Apply LoRA
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.decoder.qwen = get_peft_model(model.decoder.qwen, config)
    
    # Verify this path matches your trained weights
    weights_path = "./outputs/clip_qwen_0.8B_data_augmentation/best_metric_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()

    # 3. Batch Evaluation
    print("Evaluating targeted images...")
    results = []
    global_idx = 0
    
    with torch.no_grad():
        for imgs, _, batch_gt_captions in tqdm(dl_test, desc="Calculating METEOR", unit="batch"):
            imgs = imgs.to(device)
            
            # Batch Generation
            gen = model.generate(imgs, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
            
            for i in range(imgs.shape[0]):
                pred = tokenizer.decode(gen[i].cpu().tolist()).strip()
                gt_captions = batch_gt_captions[i]
                
                # Correctly map filename using sliced dataset
                fname = ds_test.samples[global_idx][0]
                
                score = 0.0 if not pred else meteor_metric.compute(predictions=[pred], references=[gt_captions])['meteor']
                
                results.append({
                    'fname': fname,
                    'gt_captions': gt_captions,
                    'pred': pred,
                    'score': score
                })
                global_idx += 1

    # 4. Sorting and Output
    # Sort them by ascending METEOR score
    results.sort(key=lambda x: x['score'])
    
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(results):
        print(f"\n{i+1}. Image: {item['fname']} | METEOR: {item['score'] * 100:.2f}%")
        print(f"Prediction: {item['pred']}")
        
        src_img_path = os.path.join(val_img_dir, item['fname'])
        dst_img_path = os.path.join(out_dir, item['fname'])
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

    # Save to JSON
    json_out = os.path.join(out_dir, "targeted_predictions.json")
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExtraction complete! Files saved to {out_dir}")

if __name__ == "__main__":
    main()