import os
import torch
import evaluate
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from dataset import VizWizDataset
from tokenizer import build_tokenizer
from models import CaptioningModel
from peft import LoraConfig, get_peft_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = "../datasets/vizwiz"
    
    # 1. Setup Test Dataset
    val_img_dir = os.path.join(data_root, 'val')
    val_ann = os.path.join(data_root, 'annotations', 'val.json')
    cache_dir = os.path.join(data_root, 'tokenizer_cache')
    
    print("Loading test dataset...")
    tokenizer = build_tokenizer('subword', val_ann, cache_dir)
    ds_test = VizWizDataset(val_img_dir, val_ann, tokenizer, split='test', seed=42)
    
    # Load METEOR metric
    meteor_metric = evaluate.load('meteor')

    # 2. Load 0.8B Model
    print("Loading Qwen 0.8B Model...")
    model_08b = CaptioningModel(
        encoder_name='clip', 
        decoder_type='qwen',
        decoder_model_name='Qwen/Qwen3.5-0.8B-Base',
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768 # Ensuring the correct projection dimension
    ).to(device)
    
    # --- NEW LORA WRAPPER CODE ---
    # Apply the exact same LoRA config used in train.py
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj" 
    ]
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Wrap the Qwen decoder with PEFT
    model_08b.decoder.qwen = get_peft_model(model_08b.decoder.qwen, config)
    # ------------------------------
    
    weights_path = "./outputs/clip_qwen_0.8B_clip_base/best_metric_model.pt"
    # Set strict=False just to be safe with any minor non-trainable key mismatches
    model_08b.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model_08b.eval()

    # 3. Evaluate 0.8B on Test Set (Instance-level METEOR)
    print("Evaluating 0.8B model on test set...")
    results_08b = []
    
    with torch.no_grad():
        # Added description to the existing tqdm
        for idx in tqdm(range(100), desc="0.8B Inference"):
            img_tensor, _, gt_captions = ds_test[idx]
            fname = ds_test.samples[idx][0]
            
            # Generate 0.8B caption
            img_tensor = img_tensor.unsqueeze(0).to(device)
            gen = model_08b.generate(img_tensor, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
            pred_08b = tokenizer.decode(gen[0].cpu().tolist()).strip()
            
            if not pred_08b:
                continue
                
            # Compute instance-level METEOR
            score = meteor_metric.compute(predictions=[pred_08b], references=[gt_captions])['meteor']
            
            results_08b.append({
                'idx': idx,
                'fname': fname,
                'gt_captions': gt_captions,
                'pred_08b': pred_08b,
                'score_08b': score
            })

    # Sort and get top 4
    results_08b.sort(key=lambda x: x['score_08b'], reverse=True)
    top_4 = results_08b[:4]
    
    # Sort and get top 4
    results_08b.sort(key=lambda x: x['score_08b'], reverse=True)
    top_4 = results_08b[:4]
    
    # -----------------------------------------
    # AGGRESSIVE VRAM CLEARING
    # -----------------------------------------
    print("Nuking 0.8B from orbit to free VRAM...")
    del model_08b
    
    import gc
    gc.collect()  # Force Python to clean up unreferenced objects
    torch.cuda.empty_cache() # NOW tell PyTorch to release the freed memory
    torch.cuda.reset_peak_memory_stats() # Reset fragmentation stats
    # -----------------------------------------

    # 4. Load Qwen 9B and Evaluate Top 4
    print("\nLoading Qwen 9B Model...")
    processor_9b = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B-Base", trust_remote_code=True)
    processor_9b.tokenizer.padding_side = 'left'
    if processor_9b.tokenizer.pad_token is None:
        processor_9b.tokenizer.pad_token = processor_9b.tokenizer.eos_token
        
    model_9b = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen3.5-9B-Base", 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model_9b.eval()

    print("\n" + "="*80)
    print("TOP 4 METEOR PREDICTIONS COMPARISON")
    print("="*80)
    
    text_prompt = (
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Domain: Everyday casual photography.\n"
        "Style: Simple, brief visual description. It is important not to output technical specifications or numbers.\n"
        "Image caption: "
    )

    with torch.no_grad():
        # Added tqdm here for the 9B loop
        for item in tqdm(top_4, desc="9B Inference"):
            img_path = os.path.join(val_img_dir, item['fname'])
            image = Image.open(img_path).convert("RGB")
            
            # Generate 9B caption
            inputs = processor_9b(text=[text_prompt], images=[image], return_tensors="pt", padding=True).to(device)
            generated_ids = model_9b.generate(**inputs, max_new_tokens=25)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            pred_9b = processor_9b.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            pred_9b = pred_9b.split('.')[0] + '.'
            
            # Compute 9B score
            score_9b = meteor_metric.compute(predictions=[pred_9b], references=[item['gt_captions']])['meteor']
            
            # Using tqdm.write prevents the print statements from breaking the progress bar visually
            tqdm.write(f"\nImage: {item['fname']}")
            tqdm.write(f"Ground Truths: {item['gt_captions']}")
            tqdm.write("-" * 40)
            tqdm.write(f"0.8B Pred:   {item['pred_08b']}")
            tqdm.write(f"0.8B METEOR: {item['score_08b'] * 100:.2f}%")
            tqdm.write("-" * 40)
            tqdm.write(f"9B Pred:     {pred_9b}")
            tqdm.write(f"9B METEOR:   {score_9b * 100:.2f}%")
            tqdm.write("="*80)
            
            # Prevent KV-cache buildup during loop
            del inputs, generated_ids, image 
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()