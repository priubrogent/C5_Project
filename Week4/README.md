# C5 Project: Multimodal Recognition - Image Captioning

**Master in Computer Vision Barcelona** **Module 5, Session 4**

This repository contains the codebase for Task 1 and Task 2 of the Image Captioning project. It implements a full pipeline for training, fine-tuning, and evaluating multimodal image captioning models. The architecture allows pairing frozen Vision Transformers (ViT, CLIP) with Large Language Model decoders (GPT-2, T5, SmolLM, Qwen) adapted efficiently using LoRA (Low-Rank Adaptation).

## 📁 Repository Structure

* **`train.py`**: The main training pipeline. Supports standard fine-tuning and PEFT (Parameter-Efficient Fine-Tuning) via LoRA for both encoders and decoders. Features Weights & Biases (W&B) integration, early stopping, and cosine learning rate scheduling.
* **`models.py`**: Contains the PyTorch architecture definitions. Supports dynamic coupling of multiple encoders (`ResNet`, `VGG`, `ViT`, `CLIP`) with various decoders (`GRU`, `GPT2`, `T5`, `SmolLM`, `Qwen`).
* **`dataset.py`**: Handles loading and splitting the VizWiz dataset. Includes image preprocessing (resizing, normalization) and text cleaning utilities.
* **`tokenizer.py`**: Custom tokenizer implementations supporting character-level, word-level, and BPE subword-level tokenization.
* **`evaluate_qwen.py`**: Evaluation script for large multimodal models (e.g., Qwen3.5-9B-Base). Calculates BLEU-1, BLEU-2, ROUGE-L, and METEOR metrics, and exports qualitative samples.
* **`analyze_dataset.py` & `filter_dataset.py`**: Data auditing tools used to discover and mitigate a severe dataset bias where models exploited the recurring *"Quality issues are too severe to recognize visual content"* ground truth annotations. 
* **`compare_meteor.py`**: An analytical script designed to compare predictions between fine-tuned models and zero-shot models.
* **`parameter_counter.py`**: Utility to compute the exact number of trainable (LoRA) parameters versus frozen inference parameters.

## 🛠️ Setup and Installation

1. **Environment:** Ensure you have PyTorch, Hugging Face `transformers`, `peft`, and `evaluate` installed.
2. **Dataset:** Download the VizWiz dataset and place it in the `../datasets/vizwiz` directory. The structure should be:
   ```text
   ../datasets/vizwiz/
   ├── train/
   ├── val/
   └── annotations/
       ├── train.json
       └── val.json

## 🚀 Usage Guide

### 1. Dataset Mitigation (Recommended)
To prevent the model from overfitting to unreadable images, filter the dataset to remove "Quality issues" annotations:

```bash
python filter_dataset.py
```
*Note: Add the `--filtered_annotations` flag to `train.py` to use the cleaned JSON files during training.*

### 2. Training with LoRA (Task 2)
To train a Qwen 0.8B decoder paired with a frozen CLIP encoder using a LoRA rank of 8:

```bash
python train.py \
    --encoder clip \
    --decoder qwen \
    --decoder_model_name Qwen/Qwen3.5-0.8B-Base \
    --use_lora_decoder \
    --lora_r 8 \
    --batch_size 64 \
    --epochs 12 \
    --run_name clip_qwen_0.8B_lora
```

### 3. Metric Evaluation (Zero-Shot & Multimodal)
To run a direct quantitative and qualitative evaluation using a larger, zero-shot model like Qwen3.5-9B:

```bash
python evaluate_qwen.py --model Qwen/Qwen3.5-9B-Base --batch_size 2
```

### 4. Parameter & Metric Analysis
To check the exact parameter footprint of your LoRA configuration:

```bash
python parameter_counter.py
```

To analyze METEOR discrepancies and extract the top-scoring predicted images:

```bash
python compare_meteor.py
```

## 📊 Evaluation Metrics
The pipeline automatically computes standard NLP metrics using the Hugging Face `evaluate` library:

* **BLEU (1 & 2):** Measures n-gram precision against human references.
* **ROUGE-L:** Measures the longest common subsequence (LCS) to capture sentence structure.
* **METEOR:** A metric that includes stemming and synonym matching, used to compare against our baselines.