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

NUMBER_OF_IMAGES = 50

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "../datasets/vizwiz"
    out_dir = "./outputs/lowest_meteor_images_week_5"
    batch_size = 32  
    
    # 1. Setup Test Dataset
    val_img_dir = os.path.join(data_root, 'val')
    val_ann = os.path.join(data_root, 'annotations', 'val.json') 
    cache_dir = os.path.join(data_root, 'tokenizer_cache')
    
    print("Loading test dataset...")
    tokenizer = build_tokenizer('subword', val_ann, cache_dir)
    ds_test = VizWizDataset(val_img_dir, val_ann, tokenizer, split='test', seed=42)
    
    # --- LIMIT TO FIRST 1000 ENTRIES ---
    ds_test.samples = ds_test.samples[:1000]
    print(f"Dataset limited to the first {len(ds_test.samples)} entries.")
    
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
    
    weights_path = "./outputs/clip_qwen_0.8B_data_augmentation/best_metric_model.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()

    # 3. Batch Evaluation
    print("Evaluating first 1000 entries...")
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
    results.sort(key=lambda x: x['score'])
    bottom = results[:NUMBER_OF_IMAGES]
    
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(bottom):
        print(f"\n{i+1}. Image: {item['fname']} | METEOR: {item['score'] * 100:.2f}%")
        
        src_img_path = os.path.join(val_img_dir, item['fname'])
        dst_img_path = os.path.join(out_dir, item['fname'])
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

    with open(os.path.join(out_dir, "lowest_meteor_predictions.json"), 'w') as f:
        json.dump(bottom, f, indent=4)
        
    print(f"\nExtraction complete for first 1000 entries.")

if __name__ == "__main__":
    main()