import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from dataset import VizWizDataset, collate_fn
from tokenizer import build_tokenizer
from models import CaptioningModel

BOILERPLATE_TEXT = "quality issues are too severe to recognize visual content"

def evaluate_model_boilerplate(model, dl_test, tokenizer, device, desc="Evaluating"):
    model.eval()
    
    correct_boilerplate = 0  # True Positives
    wrong_boilerplate = 0    # False Positives
    missed_boilerplate = 0   # False Negatives
    
    with torch.no_grad():
        for imgs, _, batch_gt_captions in tqdm(dl_test, desc=desc, unit="batch"):
            imgs = imgs.to(device)
            gen = model.generate(imgs, tokenizer.max_len - 1, tokenizer.sos_idx, tokenizer.eos_idx)
            
            for i in range(imgs.shape[0]):
                pred = tokenizer.decode(gen[i].cpu().tolist()).strip().lower()
                gt_captions = [gt.lower() for gt in batch_gt_captions[i]]
                
                # Check if GT contains boilerplate (at least one annotator said it)
                gt_has_boilerplate = any(BOILERPLATE_TEXT in gt for gt in gt_captions)
                # Check if model predicted boilerplate
                pred_has_boilerplate = BOILERPLATE_TEXT in pred
                
                if pred_has_boilerplate and gt_has_boilerplate:
                    correct_boilerplate += 1
                elif pred_has_boilerplate and not gt_has_boilerplate:
                    wrong_boilerplate += 1
                elif not pred_has_boilerplate and gt_has_boilerplate:
                    missed_boilerplate += 1

    return correct_boilerplate, wrong_boilerplate, missed_boilerplate

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
    
    # Optional: Limit dataset size for faster testing (e.g., ds_test.samples[:1000])
    # ds_test.samples = ds_test.samples[:1000]
    
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # 2. Base Architecture initialization
    print("\nInitializing Base Qwen 0.8B Model...")
    model = CaptioningModel(
        encoder_name='clip', 
        decoder_type='qwen',
        decoder_model_name='Qwen/Qwen3.5-0.8B-Base',
        vocab_size=tokenizer.vocab_size,
        hidden_dim=768
    ).to(device)
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.decoder.qwen = get_peft_model(model.decoder.qwen, config)

    # --- 3. Evaluate Previous Model (Standard) ---
    weights_path_1 = "./clip_qwen_0.8B/best_metric_model.pt" # Ensure this path is correct
    print(f"\nLoading Standard Model weights from: {weights_path_1}")
    model.load_state_dict(torch.load(weights_path_1, map_location=device), strict=False)
    
    standard_correct, standard_wrong, standard_missed = evaluate_model_boilerplate(
        model, dl_test, tokenizer, device, desc="Eval Standard"
    )

    # --- 4. Evaluate New Model (Data Augmented) ---
    weights_path_2 = "./outputs/clip_qwen_0.8B_data_augmentation/best_metric_model.pt" # Ensure this path is correct
    print(f"\nLoading Augmented Model weights from: {weights_path_2}")
    model.load_state_dict(torch.load(weights_path_2, map_location=device), strict=False)
    
    aug_correct, aug_wrong, aug_missed = evaluate_model_boilerplate(
        model, dl_test, tokenizer, device, desc="Eval Augmented"
    )

    # --- 5. Print Results ---
    print("\n" + "="*50)
    print("BOILERPLATE PREDICTION ANALYSIS")
    print("="*50)
    print("STANDARD MODEL (Previous Week):")
    print(f"  - Good Predictions (True Positives):  {standard_correct}")
    print(f"  - Wrong Predictions (False Positives): {standard_wrong}")
    print(f"  - Missed Boilerplates (False Negatives): {standard_missed}")
    print("\nAUGMENTED MODEL (This Week):")
    print(f"  - Good Predictions (True Positives):  {aug_correct}")
    print(f"  - Wrong Predictions (False Positives): {aug_wrong}")
    print(f"  - Missed Boilerplates (False Negatives): {aug_missed}")
    print("="*50)

    # --- 6. Plotting ---
    labels = ['Standard Model', 'Augmented Model']
    correct_vals = [standard_correct, aug_correct]
    wrong_vals = [standard_wrong, aug_wrong]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, correct_vals, width, label='Good Predicted (GT matches)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, wrong_vals, width, label='Wrong Predicted (GT does not match)', color='#d62728')

    ax.set_ylabel('Number of Images')
    ax.set_title('Impact of Data Augmentation on Boilerplate Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('boilerplate_comparison.png', dpi=300)
    print("\nPlot saved successfully as 'boilerplate_comparison.png'")

if __name__ == "__main__":
    main()