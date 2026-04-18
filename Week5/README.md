# VizWiz Robust Image Captioning & Augmentation Pipeline

This repository contains a complete Vision-Language pipeline designed to train, augment, and evaluate image captioning models on the VizWiz dataset. The project specifically tackles challenges inherent to photographs taken by visually impaired users, addressing issues like label imbalance (prior shift) and long-tail object distribution.

## 📂 Repository Structure

### 🧠 Core Architecture & Training
* **`train.py`**: The main training script. Handles data loading, optimizer scheduling (with cosine warmup for Qwen), early stopping, and Weights & Biases (W&B) logging. Supports loading pretrained encoder weights and selectively freezing layers.
* **`models.py`**: Defines the `CaptioningModel` class. Contains wrappers for CNNs, ViT, and CLIP encoders, along with decoders ranging from standard RNNs (GRU/LSTM with additive attention) to transformer-based LLMs (Qwen3.5, SmolLM, GPT-2, T5).
* **`dataset.py`**: PyTorch `Dataset` implementation for VizWiz. Safely partitions original VizWiz images for the validation split while allowing synthetic/augmented images strictly in the training split to prevent data leakage.
* **`tokenizer.py`**: Custom tokenization factory supporting `char`, `word`, and Hugging Face `subword` (BPE) text representations.

### 🛠️ Data Augmentation & Synthesis
* **`transform_images.py`**: Simulates typical VizWiz camera artifacts. Downscales high-quality images and applies algorithmic degradations (directional motion blur, JPEG compression artifacts, and overexposure) to create "moderate" and "severe" image variants.
* **`generate_sample.py`**: A lightweight utility to generate and save local samples of the original, moderate, and severe augmentations for quick visual inspection before running bulk processing.
* **`caption_generation.py`**: Uses the zero-shot capabilities of `Qwen/Qwen3.5-9B-Base` to auto-generate ground-truth captions for synthetic images, forcing the model to include specific objects (from `image_metadata.json`) in the very first sentence.
* **`merge_datasets.py`**: Safely merges the newly generated synthetic images and their COCO-formatted JSON annotations into the original VizWiz training dataset, including strict deduplication checks.
* **`image_metadata.json`**: A dictionary containing the target objects and prompt contexts used to guide the synthetic caption generation for 1,000 baseline images.

### 📊 Evaluation & Bias Analysis
* **`analyze_imbalance.py`**: Uses `spaCy` to extract nouns from captions and plots the long-tail distribution of objects in the VizWiz dataset. It also tracks linguistic biases, such as the frequency of hedge phrases (e.g., "too blurry").
* **`evaluate_imbalance.py`**: Evaluates model performance (METEOR) separately on images containing common "Head" objects versus rare "Tail" objects to quantify semantic bias.
* **`analyze_boilerplate.py`**: Quantifies overfitting to unanswerable images by counting True Positives (Good) and False Positives (Wrong) of the boilerplate phrase: *"Quality issues are too severe to recognize visual content"*. Generates comparison bar charts between standard and augmented models.
* **`worst_performance.py`**: Evaluates the test set, isolates a specified number of images with the lowest METEOR scores, and exports them to a dedicated folder for qualitative failure analysis.
* **`targeted_worst_performance.py`**: Similar to the above, but strictly evaluates performance on a hardcoded list of target images to track improvements across different model iterations.

---

## 🚀 Usage Guide

### 1. Data Augmentation
To simulate camera artifacts on your clean dataset and generate COCO-style captions:
```bash
# 1. Generate synthetic captions using Qwen 9B
python caption_generation.py

# 2. Apply visual degradations (blur, overexposure)
python transform_images.py

# 3. Merge with the official VizWiz dataset
python merge_datasets.py
```

### 2. Training

To train a Qwen 0.8B decoder with a frozen CLIP encoder using the augmented dataset:

```bash
python train.py \
    --encoder clip \
    --decoder qwen \
    --decoder_model_name Qwen/Qwen3.5-0.8B-Base \
    --augmented_annotations \
    --use_lora_decoder \
    --freeze_encoder \
    --epochs 20 \
    --batch_size 32 \
    --run_name clip_qwen_0.8B_augmented
```

### 3. Analysis and evaluation

To analyze if your model is overfitting to the "unanswerable" boilerplate text:
```bash
python analyze_boilerplate.py
```

To evaluate how well your model performs on rare (tail) objects vs. common (head) objects:
```bash
python evaluate_imbalance.py --model_weights ./outputs/your_run/best_metric_model.pt
```

To extract the worst-performing predictions for your qualitative report:
```bash
python worst_performance.py
```

## Supported Metrics

During evaluation and early stopping, the pipeline computes standard NLP matching metrics using the Hugging Face evaluate library:
- BLEU (1 & 2): N-gram precision.
- ROUGE-L: Longest common subsequence for sentence structure.
- METEOR: Includes stemming and synonym matching, robust for descriptive captions.