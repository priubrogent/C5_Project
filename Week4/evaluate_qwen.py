import torch
import json
import os
import argparse
import random
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import evaluate

from dataset import VizWizDataset
from tokenizer import build_tokenizer

def get_fixed_examples(dataset, n=10, seed=42):
    """Select n fixed indices from a dataset for qualitative evaluation."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    return indices

def evaluate_qwen_multimodal(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {args.model} on {device}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Setup paths matching train.py
    train_img_dir = os.path.join(args.data_root, 'train')
    val_img_dir   = os.path.join(args.data_root, 'val')
    train_ann     = os.path.join(args.data_root, 'annotations', 'train_filtered.json')
    val_ann       = os.path.join(args.data_root, 'annotations', 'val_filtered.json')
    cache_dir     = os.path.join(args.data_root, 'tokenizer_cache')

    # Instantiate tokenizer and datasets using VizWizDataset
    print("Loading datasets via VizWiz class...")
    tokenizer = build_tokenizer('subword', train_ann, cache_dir)
    
    ds_val = VizWizDataset(train_img_dir, train_ann, tokenizer,
                           split='val', val_fraction=args.val_fraction, seed=args.seed)
    ds_test = VizWizDataset(val_img_dir, val_ann, tokenizer, 
                            split='test', seed=args.seed)

    # Get the images for qualitative results
    val_qual_indices = get_fixed_examples(ds_val, n=100, seed=args.seed)
    wandb_indices = val_qual_indices[:10]

    # ---------------------------------------------------------
    # PART 1: QUALITATIVE EVALUATION (W&B 10 Images)
    # ---------------------------------------------------------
    print(f"\n--- Generating Qualitative Results (W&B 10 images) ---")
    qualitative_results = []
    
    with torch.no_grad():
        for idx in tqdm(wandb_indices):
            fname, gt_captions = ds_val.samples[idx]
            img_path = os.path.join(train_img_dir, fname)
            image = Image.open(img_path).convert("RGB")
            
            # 1. Add the explicit Qwen image tokens before the text prompt
            text_prompt = (
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                "Domain: Everyday casual photography.\n"
                "Style: Simple, brief visual description. It is important not to output technical specifications or numbers.\n"
                "Image caption: "
            )
            
            # 2. Pass the raw string directly into the processor
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt", padding=True).to(device)
            
            # 3. Generate
            generated_ids = model.generate(**inputs, max_new_tokens=25)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
            
            # 4. Post-processing
            pred = raw_pred.strip()
            pred = pred.split('.')[0] + '.'
            
            qualitative_results.append({
                "image_id": idx,
                "file_name": fname,
                "prediction": pred,
                "references": gt_captions
            })
            
    qualitative_path = os.path.join(args.out_dir, "qwen_qualitative_samples_filtered.json")
    with open(qualitative_path, "w") as f:
        json.dump(qualitative_results, f, indent=4)
    print(f"Saved Qualitative results to: {qualitative_path}")


    # ---------------------------------------------------------
    # PART 2: QUANTITATIVE EVALUATION
    # ---------------------------------------------------------
    print(f"\n--- Running Quantitative Evaluation on Test Set ---")
    predictions, references = [], []
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(ds_test.samples), batch_size)):
            batch = ds_test.samples[i:i+batch_size]
            images, texts, batch_refs = [], [], []

            for fname, caps in batch:
                img_path = os.path.join(val_img_dir, fname)
                images.append(Image.open(img_path).convert("RGB"))
                batch_refs.append(caps)
                
                text_prompt = (
                    "<|vision_start|><|image_pad|><|vision_end|>\n"
                    "Domain: Everyday casual photography.\n"
                    "Style: Simple, brief visual description. It is important not to output technical specifications or numbers.\n"
                    "Image caption: "
                )
                texts.append(text_prompt)
            
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=25)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            
            for out_text, refs in zip(output_texts, batch_refs):
                pred = out_text.strip()
                pred = pred.split('.')[0] + '.'
                
                predictions.append(pred)
                references.append(refs)

    # Compute final metrics
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    bleu1 = bleu.compute(predictions=predictions, references=references, max_order=1)['bleu'] * 100
    bleu2 = bleu.compute(predictions=predictions, references=references, max_order=2)['bleu'] * 100
    rougeL = rouge.compute(predictions=predictions, references=[r[0] for r in references])['rougeL'] * 100
    met = meteor.compute(predictions=predictions, references=references)['meteor'] * 100
    
    results_str = (
        f"DIRECT EVALUATION: {args.model}\n"
        f"TEST SAMPLES: {len(ds_test.samples)}\n"
        f"BLEU-1:  {bleu1:.2f}%\n"
        f"BLEU-2:  {bleu2:.2f}%\n"
        f"ROUGE-L: {rougeL:.2f}%\n"
        f"METEOR:  {met:.2f}%\n"
    )
    
    print("\n" + "="*50)
    print(results_str.strip())
    print("="*50)
    
    metrics_path = os.path.join(args.out_dir, "qwen_metrics_filtered.txt")
    with open(metrics_path, "w") as f:
        f.write(results_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B-Base")
    parser.add_argument("--data_root", type=str, default="../datasets/vizwiz")
    parser.add_argument("--out_dir", type=str, default="./outputs/qwen_eval")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()
    
    evaluate_qwen_multimodal(args)