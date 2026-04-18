# VizWiz Robust Image Captioning & Augmentation Pipeline

This repository contains a complete Vision-Language pipeline designed to train, augment, and evaluate image captioning models on the VizWiz dataset. The project specifically tackles challenges inherent to photographs taken by visually impaired users, addressing issues like label imbalance (prior shift) and long-tail object distribution.

## 📂 Repository Structure

### 🧠 Core Architecture & Training
* **`train.py`**: The main training script. Handles data loading, optimizer scheduling (with cosine warmup for Qwen), early stopping, and Weights & Biases (W&B) logging. Supports loading pretrained encoder weights and selectively freezing layers.
* **`models.py`**: Defines the `CaptioningModel` class. Contains wrappers for CNNs, ViT, and CLIP encoders, along with decoders ranging from standard RNNs (GRU/LSTM with additive attention) to transformer-based LLMs (Qwen3.5, SmolLM, GPT-2, T5).
* **`dataset.py`**: PyTorch `Dataset` implementation for VizWiz. Safely partitions original VizWiz images for the validation split while allowing synthetic/augmented images strictly in the training split to prevent data leakage.
* **`tokenizer.py`**: Custom tokenization factory supporting `char`, `word`, and Hugging Face `subword` (BPE) text representations.

### 🛠️ Data Augmentation & Synthesis

#### Generative Synthesis with Stable Diffusion
* **`image_generation/stable_diffusion_base.py`**: Baseline exploration script for evaluating different Stable Diffusion model variants (SD 2.1, SDXL, SD Turbo). Tests multiple prompt templates designed to capture first-person perspectives and typical VizWiz characteristics (low lighting, occlusions, awkward angles). Tracks inference times and generates samples across all model variants to determine the best baseline model for synthetic image generation.
* **`image_generation/stable_diffusion_exploration.py`**: Hyperparameter exploration and configuration tuning script for the selected Stable Diffusion model. Systematically tests different schedulers (DDIM, DDPM), negative prompts, classifier-free guidance (CFG) scales, and inference steps. Integrates with Weights & Biases (W&B) for tracking experiments and comparing results across hyperparameter combinations.
* **`image_generation/stable_diffusion_final_generation.py`**: Production script for generating the complete 1,000 synthetic images used for data augmentation. Uses the optimal model configuration determined through exploration (e.g., SDXL with DDIM scheduler, CFG 7.5, and 30 inference steps). Generates diverse first-person perspectives depicting common household objects in indoor settings.
* **`image_metadata.json`**: A dictionary containing the target objects and prompt contexts used to guide synthetic image generation for 1,000 baseline images.

#### Visual Degradation of Generated Images
* **`transform_images.py`**: Applies typical VizWiz camera artifacts to the generated synthetic images to create realistic degradation. Applies algorithmic degradations including directional motion blur, JPEG compression artifacts, and overexposure to create "moderate" and "severe" quality variants that the model learns to caption despite quality issues.
* **`generate_sample.py`**: A lightweight utility to generate and save local samples of the original, moderate, and severe image variants for quick visual inspection before running bulk processing on all 1,000 generated images.
* **`caption_generation.py`**: Uses the zero-shot capabilities of `Qwen/Qwen3.5-9B-Base` to auto-generate ground-truth captions for the degraded synthetic images, forcing the model to include specific target objects (from `image_metadata.json`) in the very first sentence.
* **`merge_datasets.py`**: Safely merges the newly generated and degraded synthetic images along with their COCO-formatted JSON annotations into the original VizWiz training dataset, including strict deduplication checks.

### 📊 Evaluation & Bias Analysis
* **`analyze_imbalance.py`**: Uses `spaCy` to extract nouns from captions and plots the long-tail distribution of objects in the VizWiz dataset. It also tracks linguistic biases, such as the frequency of hedge phrases (e.g., "too blurry").
* **`evaluate_imbalance.py`**: Evaluates model performance (METEOR) separately on images containing common "Head" objects versus rare "Tail" objects to quantify semantic bias.
* **`analyze_boilerplate.py`**: Quantifies overfitting to unanswerable images by counting True Positives (Good) and False Positives (Wrong) of the boilerplate phrase: *"Quality issues are too severe to recognize visual content"*. Generates comparison bar charts between standard and augmented models.
* **`worst_performance.py`**: Evaluates the test set, isolates a specified number of images with the lowest METEOR scores, and exports them to a dedicated folder for qualitative failure analysis.
* **`targeted_worst_performance.py`**: Similar to the above, but strictly evaluates performance on a hardcoded list of target images to track improvements across different model iterations.

---

## 🚀 Usage Guide

### 1. Data Augmentation
The complete augmentation pipeline involves generating synthetic images, degrading them to simulate real-world VizWiz conditions, captioning them, and finally merging into the training dataset:

#### Step 1: Generate Synthetic Images (Stable Diffusion)
```bash
# 1a. (Optional) Test different model variants and prompt templates
python image_generation/stable_diffusion_base.py

# 1b. (Optional) Explore hyperparameter configurations with W&B tracking
python image_generation/stable_diffusion_exploration.py

# 1c. Generate the final 1,000 synthetic images using optimal configuration
python image_generation/stable_diffusion_final_generation.py
```

#### Step 2: Degrade Generated Images and Caption Them
```bash
# 2a. Apply visual degradations (blur, overexposure, etc.) to generated images
python transform_images.py

# 2b. Generate synthetic captions using Qwen 9B
python caption_generation.py

# 2c. Merge the augmented images and captions with the official VizWiz dataset
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