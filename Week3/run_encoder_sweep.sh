#!/usr/bin/env bash
# ============================================================
# run_encoder_sweep.sh
# Trains the baseline captioning model with every encoder.
# Baseline config: GRU decoder, 1 layer, 512 dim, char tokenizer,
#                  teacher forcing, Adam, 50 epochs.
#
# Usage:
#   bash run_encoder_sweep.sh [OPTIONS]
#
# Options:
#   --data_root DIR       Path to vizwiz_dataset   (default below)
#   --out_root  DIR       Path to output root       (default below)
#   --wandb_entity NAME   W&B entity / username     (required for wandb)
#   --wandb_project NAME  W&B project name          (default: c5-week3-captioning)
#   --epochs N            Epochs per run            (default: 50)
#   --batch_size N        Batch size                (default: 64)
#   --device DEVICE       e.g. cuda / cpu           (default: cuda)
#   --no_wandb            Disable W&B logging
# ============================================================
set -euo pipefail

# ---------- defaults ----------
DATA_ROOT="/media/arnau-marcos-almansa/Ubuntu Data/vizwiz_dataset"
OUT_ROOT="/home/arnau-marcos-almansa/workspace/C5_Project/Week3/outputs"
WANDB_ENTITY="just-an-arbitrary-team-name"
WANDB_PROJECT="mcv-c5-image_captioning"
EPOCHS=50
BATCH_SIZE=64
DEVICE="cuda"
USE_WANDB=true

# ---------- parse CLI ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)     DATA_ROOT="$2";     shift 2 ;;
        --out_root)      OUT_ROOT="$2";      shift 2 ;;
        --wandb_entity)  WANDB_ENTITY="$2";  shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --device)        DEVICE="$2";        shift 2 ;;
        --no_wandb)      USE_WANDB=false;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


ENCODERS=(resnet18 resnet34 resnet50 vgg16 vgg19)

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/sweep_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/summary.csv"

# CSV header
echo "encoder,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Encoder sweep started"
log "  encoders    : ${ENCODERS[*]}"
log "  epochs      : $EPOCHS"
log "  batch_size  : $BATCH_SIZE"
log "  device      : $DEVICE"
log "  data_root   : $DATA_ROOT"
log "  out_root    : $OUT_ROOT"
log "  wandb       : $USE_WANDB (entity=$WANDB_ENTITY, project=$WANDB_PROJECT)"
log "========================================================"

for ENCODER in "${ENCODERS[@]}"; do
    RUN_NAME="baseline_${ENCODER}"
    RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log ""
    log "-------- Starting run: $RUN_NAME --------"

    WANDB_ARGS=()
    if $USE_WANDB; then
        WANDB_ARGS=(--wandb --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT")
    fi

    set +e
    python train.py \
        --encoder       "$ENCODER" \
        --decoder       gru \
        --decoder_layers 1 \
        --hidden_dim    512 \
        --embed_dim     512 \
        --dropout       0.0 \
        --text_repr     char \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH_SIZE" \
        --lr            1e-3 \
        --weight_decay  1e-4 \
        --optimizer     adam \
        --teacher_forcing \
        --lr_decay      0.5 \
        --lr_patience   5 \
        --run_name      "$RUN_NAME" \
        --seed          42 \
        --num_workers   4 \
        --data_root     "$DATA_ROOT" \
        --out_root      "$OUT_ROOT" \
        --device        "$DEVICE" \
        "${WANDB_ARGS[@]}" \
        2>&1 | tee "$RUN_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"
    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        # Parse test_results.json with python (guaranteed available)
        ROW=$(python - <<EOF
import json, sys
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${ENCODER},${ROW}" >> "$SUMMARY_CSV"
        log "  -> $RUN_NAME finished OK  |  $ROW"
    else
        echo "${ENCODER},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  -> $RUN_NAME FAILED (exit=$EXIT_CODE). See $RUN_LOG"
    fi
done

log ""
log "========================================================"
log "Sweep complete. Summary:"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log "Full logs: $SWEEP_LOG_DIR"
log "========================================================"
