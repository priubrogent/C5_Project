import argparse
import json
import os
import re
import shutil  # Added for file copying
import torch
import evaluate as hf_evaluate
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from dataset import VizWizDataset, collate_fn
from models import CaptioningModel
from tokenizer import build_tokenizer

def load_word_list(filepath, has_counts=False):
    """Loads words from a text file, cleanly ignoring tags and counts."""
    words = set()
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File not found -> {filepath}")
        return words

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if ']' in line:
                line = line.split(']')[-1].strip()
                
            if not line:
                continue

            if has_counts:
                if ':' in line:
                    word = line.split(':')[0].strip()
                else:
                    word = line.strip()
            else:
                word = line.strip()

            if word and not word.isdigit():
                words.add(word.lower())
                
    return set(words)

def compute_meteor(meteor_metric, predictions, references):
    if not predictions:
        return 0.0
    return meteor_metric.compute(predictions=predictions, references=references)['meteor'] * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weights', type=str, default='./clip_qwen_0.8B/best_metric_model.pt')
    parser.add_argument('--head_words_file', type=str, default='head_objects.txt')
    parser.add_argument('--tail_words_file', type=str, default='rare_objects_sample.txt')
    parser.add_argument('--data_root', type=str, default='../datasets/vizwiz')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--output_file', type=str, default='imbalance_results.txt', help='TXT file to save the report.')
    parser.add_argument('--save_tail_preds', action='store_true', help='Saves the predictions of the "Tail Only" subset in the text file.')
    parser.add_argument('--fast_tail_eval', action='store_true', help='Filters out non-tail images before running GPU inference to save time.')

    parser.add_argument('--encoder', default='clip')
    parser.add_argument('--decoder', default='qwen')
    parser.add_argument('--decoder_model_name', default='Qwen/Qwen3.5-0.8B-Base')
    parser.add_argument('--text_repr', default='subword')
    parser.add_argument('--use_lora_decoder', action='store_true', default=True)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("\n" + "="*50)
    print(" ⚙️  SYSTEM SETUP")
    print("="*50)
    if device.type == 'cuda':
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ WARNING: No GPU detected. Using CPU. Inference will be slow!")

    print("\nLoading vocabulary files...")
    head_words = load_word_list(args.head_words_file, has_counts=True)
    tail_words = load_word_list(args.tail_words_file, has_counts=False)
    
    if head_words and tail_words:
        print(f"✅ Loaded {len(head_words)} HEAD words and {len(tail_words)} TAIL words.")
    else:
        print("\n❌ Missing vocabulary files. Exiting.")
        return

    val_ann = os.path.join(args.data_root, 'annotations', 'val.json')
    val_img_dir = os.path.join(args.data_root, 'val')
    cache_dir = os.path.join(args.data_root, 'tokenizer_cache')
    train_ann = os.path.join(args.data_root, 'annotations', 'train.json')
    
    print("\nBuilding tokenizer and loading dataset...")
    tokenizer = build_tokenizer(args.text_repr, train_ann, cache_dir)
    ds_test = VizWizDataset(val_img_dir, val_ann, tokenizer, split='test')
    
    if args.fast_tail_eval:
        print("\n✂️  [Fast Eval] Filtering dataset to keep ONLY images with TAIL words...")
        filtered_samples = []
        for fname, captions in ds_test.samples:
            is_tail_image = False
            for cap in captions:
                clean_cap = re.sub(r'[^\w\s]', '', cap.lower())
                ref_words = set(clean_cap.split())
                if not tail_words.isdisjoint(ref_words):
                    is_tail_image = True
                    break 
            
            if is_tail_image:
                filtered_samples.append((fname, captions))
                
        ds_test.samples = filtered_samples
        print(f"✅ Dataset drastically reduced to {len(ds_test.samples)} TAIL images.")

    dl_test = DataLoader(ds_test, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)

    model = CaptioningModel(
        encoder_name=args.encoder,
        decoder_type=args.decoder,
        vocab_size=tokenizer.vocab_size,
        decoder_model_name=args.decoder_model_name,
        hidden_dim=768, 
    ).to(device)

    if args.use_lora_decoder and args.decoder == 'qwen':
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
        )
        model.decoder.qwen = get_peft_model(model.decoder.qwen, config)

    print(f"\nLoading model weights from {args.model_weights}...")
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval()

    meteor_metric = hf_evaluate.load('meteor')

    head_preds, head_refs = [], []
    tail_preds, tail_refs = [], []
    both_preds, both_refs = [], []
    tail_only_preds, tail_only_refs = [], []
    tail_only_details = []
    
    # State tracking for image saving
    saved_first_tail_image = False
    global_img_idx = 0
    
    print("\nStarting evaluation...")
    
    with torch.no_grad():
        for imgs, _, all_captions in tqdm(dl_test, desc="Generating Captions", unit="batch"):
            imgs = imgs.to(device)
            gen = model.generate(imgs, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
            
            for i in range(imgs.shape[0]):
                pred = tokenizer.decode(gen[i].cpu().tolist())
                refs = all_captions[i]
                
                ref_words = set()
                for cap in refs:
                    clean_cap = re.sub(r'[^\w\s]', '', cap.lower())
                    ref_words.update(clean_cap.split())
                
                is_head = not head_words.isdisjoint(ref_words)
                is_tail = not tail_words.isdisjoint(ref_words)
                
                if is_head:
                    head_preds.append(pred)
                    head_refs.append(refs)
                
                if is_tail:
                    tail_preds.append(pred)
                    tail_refs.append(refs)
                    
                    # Fixed: Moved the details append inside the correct block
                    if args.save_tail_preds:
                        tail_only_details.append({
                            "pred": pred,
                            "gt": refs
                        })
                        
                        # Image Saving Logic
                        if not saved_first_tail_image:
                            # 1. Get the original filename from the dataset
                            original_fname = ds_test.samples[global_img_idx][0]
                            src_img_path = os.path.join(val_img_dir, original_fname)
                            
                            # 2. Define where to save it (same directory as output txt)
                            out_dir = os.path.dirname(os.path.abspath(args.output_file))
                            if not out_dir:
                                out_dir = "."
                            dst_img_path = os.path.join(out_dir, "first_tail_only_image.jpg")
                            
                            # 3. Copy the file
                            try:
                                shutil.copy2(src_img_path, dst_img_path)
                                print(f"\n📸 Saved first Tail-Only image to: {dst_img_path}")
                                saved_first_tail_image = True
                            except Exception as e:
                                print(f"\n⚠️ Failed to copy image: {e}")
                    
                if is_head and is_tail:
                    both_preds.append(pred)
                    both_refs.append(refs)
                    
                if is_tail and not is_head:
                    tail_only_preds.append(pred)
                    tail_only_refs.append(refs)
                                
                # Increment the global index counter
                global_img_idx += 1

    print("\nComputing METEOR scores...")
    head_score = compute_meteor(meteor_metric, head_preds, head_refs)
    tail_score = compute_meteor(meteor_metric, tail_preds, tail_refs)
    both_score = compute_meteor(meteor_metric, both_preds, both_refs)
    tail_only_score = compute_meteor(meteor_metric, tail_only_preds, tail_only_refs)

    report = []
    report.append("="*60)
    report.append(" 📊 SEMANTIC DISTRIBUTION EVALUATION (METEOR)")
    if args.fast_tail_eval:
        report.append(" ⚠️ [FAST EVAL MODE: ONLY TAIL IMAGES PROCESSED]")
    report.append("="*60)
    
    if not args.fast_tail_eval:
        report.append(f"1. HEAD Objects Subset ({len(head_preds)} images):")
        report.append(f"   METEOR Score: {head_score:.2f}%\n")
    else:
        report.append(f"1. HEAD Objects Subset (Skipped due to --fast_tail_eval)\n")
        
    report.append(f"2. TAIL Objects Subset (Total) ({len(tail_preds)} images):")
    report.append(f"   METEOR Score: {tail_score:.2f}%\n")
    
    report.append(f"3. INTERSECTION (Both Head & Tail) ({len(both_preds)} images):")
    report.append(f"   METEOR Score: {both_score:.2f}%\n")
    
    report.append(f"4. TAIL ONLY (No Head Words) ({len(tail_only_preds)} images):")
    report.append(f"   METEOR Score: {tail_only_score:.2f}%")
    report.append("="*60)
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(report_text + "\n\n")
        
        if args.save_tail_preds:
            f.write("="*60 + "\n")
            f.write(" 📝 TAIL ONLY SUBSET PREDICTIONS\n")
            f.write("="*60 + "\n\n")
            
            for idx, detail in enumerate(tail_only_details, 1):
                f.write(f"--- Example {idx} ---\n")
                f.write(f"Prediction : {detail['pred']}\n")
                f.write("Ground Truths:\n")
                for gt in detail['gt']:
                    f.write(f"  - {gt}\n")
                f.write("\n")
                
    print(f"\n📁 Results and report successfully saved to: {args.output_file}")

if __name__ == '__main__':
    main()