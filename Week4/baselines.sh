#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python train.py --encoder vit-b-16 --decoder t5 --run_name baseline_vit-b-16_t5 
python train.py --encoder clip --decoder gpt2 --run_name baseline_clip_gpt2 
python train.py --encoder clip --decoder smollm --run_name baseline_clip_smollm 
python train.py --encoder vit-b-32 --decoder gpt2 --run_name baseline_vit-b-32_gpt2
python train.py --encoder vit-b-32 --decoder t5 --run_name baseline_vit-b-32_t5 

