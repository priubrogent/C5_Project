import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import TRAIN_SEQS, VAL_SEQS

# --- Configuration ---
ANNOTATIONS_DIR = "/ghome/mcv/datasets/C5/KITTI-MOTS/instances_txt"
OUTPUT_DIR = "./dataset_stats/"

# Font size settings for the plot
FONT_SIZE_TITLE = 24
FONT_SIZE_AXIS = 20
FONT_SIZE_TICKS = 18
FONT_SIZE_LEGEND = 18

def generate_dataset_report():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def analyze_split(sequences):
        stats = {"cars": 0, "peds": 0, "ignore": 0, "frames": 0}
        
        for i in tqdm(sequences, desc="Analyzing Sequences"):
            txt_path = Path(ANNOTATIONS_DIR) / f"{i:04d}.txt"
            if not txt_path.exists(): continue
            
            frames_in_seq = set()
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    frame_idx, obj_id = int(parts[0]), int(parts[1])
                    # KITTI-MOTS Class ID extraction logic
                    class_id = obj_id // 1000 
                    
                    frames_in_seq.add(frame_idx)
                    if class_id == 1: stats["cars"] += 1
                    elif class_id == 2: stats["peds"] += 1
                    elif class_id == 10: stats["ignore"] += 1
            
            stats["frames"] += len(frames_in_seq)
        return stats

    print("Gathering Statistics...")
    train_stats = analyze_split(TRAIN_SEQS)
    val_stats = analyze_split(VAL_SEQS)

    # --- Create Bar Graphic ---
    labels = ['Cars', 'Pedestrians', 'Ignore']
    train_vals = [train_stats["cars"], train_stats["peds"], train_stats["ignore"]]
    val_vals = [val_stats["cars"], val_stats["peds"], val_stats["ignore"]]

    x = np.arange(len(labels))
    width = 0.35

    # Increased figure size for better resolution with large fonts
    fig, ax = plt.subplots(figsize=(12, 8))
    
    rects1 = ax.bar(x - width/2, train_vals, width, label='Train Split', color='skyblue')
    rects2 = ax.bar(x + width/2, val_vals, width, label='Val Split', color='salmon')

    # Styling with larger fonts
    ax.set_ylabel('Total Instance Count', fontsize=FONT_SIZE_AXIS, labelpad=15)
    ax.set_title('KITTI-MOTS Instance Distribution', fontsize=FONT_SIZE_TITLE, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE_TICKS)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels on top of bars for extra clarity
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution_high_res.png"), dpi=300)
    print(f"Graphic saved to {OUTPUT_DIR}class_distribution_high_res.png")

    # --- Final Text Summary ---
    print(f"\n--- DATASET SUMMARY ---")
    print(f"Train: {train_stats['frames']} frames, {train_stats['cars']} cars, {train_stats['peds']} peds")
    print(f"Val:   {val_stats['frames']} frames, {val_stats['cars']} cars, {val_stats['peds']} peds")

if __name__ == "__main__":
    generate_dataset_report()