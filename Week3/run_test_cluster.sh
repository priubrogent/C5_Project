#!/bin/bash
#SBATCH -n 8
#SBATCH -N 1
#SBATCH -D /hhome/priubrogent/mcvpol/C5/Week3/
#SBATCH -p dcca40
#SBATCH --mem 35048
#SBATCH -o %x_%u_%j.out
#SBATCH -e %x_%u_%j.err
#SBATCH --gres gpu:1

# Quick test run to verify cluster setup and qualitative wandb logging.

SCRIPT=/hhome/priubrogent/mcvpol/C5/Week3/train.py

python $SCRIPT \
    --encoder resnet18 \
    --decoder gru \
    --decoder_layers 1 \
    --hidden_dim 256 \
    --embed_dim 256 \
    --dropout 0.0 \
    --text_repr char \
    --epochs 3 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --optimizer adam \
    --teacher_forcing \
    --lr_decay 0.5 \
    --lr_patience 2 \
    --es_patience 5 \
    --es_metric val_loss \
    --run_name test_cluster \
    --seed 42 \
    --num_workers 4 \
    --data_root /hhome/priubrogent/mcvpol/vizwiz_dataset \
    --out_root /hhome/priubrogent/mcvpol/C5/Week3/outputs \
    --device cuda \
    --wandb \
    --wandb_entity just-an-arbitrary-team-name \
    --wandb_project mcv-c5-image_captioning
