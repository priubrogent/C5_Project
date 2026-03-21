#!/usr/bin/env bash
# ============================================================
# run_encoder_sweep_parallel.sh
# Runs the baseline encoder sweep across 3 GPUs in parallel.
# Encoders are batched: 3 run simultaneously, then the last 2.
#
# Usage:
#   bash run_encoder_sweep_parallel.sh [OPTIONS]
#
# Options:
#   --data_root DIR       Path to vizwiz_dataset   (default below)
#   --out_root  DIR       Path to output root       (default below)
#   --wandb_entity NAME   W&B entity / username
#   --wandb_project NAME  W&B project name          (default: mcv-c5-image_captioning)
#   --epochs N            Epochs per run            (default: 50)
#   --batch_size N        Batch size                (default: 64)
#   --gpu_ids IDS         Comma-separated GPU ids   (default: 0,1,2)
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
GPU_IDS="0,1,2"
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
        --gpu_ids)       GPU_IDS="$2";       shift 2 ;;
        --no_wandb)      USE_WANDB=false;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Parse GPU ids into an array
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

ENCODERS=(resnet18 resnet34 resnet50 vgg16 vgg19)

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/sweep_parallel_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/summary_parallel.csv"

echo "encoder,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Parallel encoder sweep started"
log "  encoders    : ${ENCODERS[*]}"
log "  epochs      : $EPOCHS"
log "  batch_size  : $BATCH_SIZE"
log "  GPUs        : ${GPUS[*]}"
log "  data_root   : $DATA_ROOT"
log "  out_root    : $OUT_ROOT"
log "  wandb       : $USE_WANDB (entity=$WANDB_ENTITY, project=$WANDB_PROJECT)"
log "========================================================"

# Launch one encoder run in the background on a given GPU.
# Sets CUDA_VISIBLE_DEVICES so the process sees exactly one GPU as cuda:0.
launch_run() {
    local ENCODER="$1"
    local GPU_ID="$2"
    local RUN_NAME="baseline_${ENCODER}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log "  Launching $RUN_NAME on GPU $GPU_ID -> $RUN_LOG"

    local WANDB_ARGS=()
    if $USE_WANDB; then
        WANDB_ARGS=(--wandb --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --encoder        "$ENCODER" \
        --decoder        gru \
        --decoder_layers 1 \
        --hidden_dim     512 \
        --embed_dim      512 \
        --dropout        0.0 \
        --text_repr      char \
        --epochs         "$EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --lr             1e-3 \
        --weight_decay   1e-4 \
        --optimizer      adam \
        --teacher_forcing \
        --lr_decay       0.5 \
        --lr_patience    5 \
        --run_name       "$RUN_NAME" \
        --seed           42 \
        --num_workers    4 \
        --max_eval_samples 2000 \
        --data_root      "$DATA_ROOT" \
        --out_root       "$OUT_ROOT" \
        --device         cuda \
        "${WANDB_ARGS[@]}" \
        > "$RUN_LOG" 2>&1
}

# Collect results for one encoder into the summary CSV.
collect_result() {
    local ENCODER="$1"
    local EXIT_CODE="$2"
    local RESULTS_FILE="${OUT_ROOT}/baseline_${ENCODER}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${ENCODER},${ROW}" >> "$SUMMARY_CSV"
        log "  $ENCODER -> OK  |  $ROW"
    else
        echo "${ENCODER},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $ENCODER -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/baseline_${ENCODER}.log"
    fi
}

# Run encoders in batches of NUM_GPUS
i=0
while [[ $i -lt ${#ENCODERS[@]} ]]; do
    PIDS=()
    BATCH_ENCODERS=()

    # Launch up to NUM_GPUS runs in parallel
    for (( g=0; g<NUM_GPUS && i<${#ENCODERS[@]}; g++, i++ )); do
        ENCODER="${ENCODERS[$i]}"
        GPU_ID="${GPUS[$g]}"
        BATCH_ENCODERS+=("$ENCODER")
        launch_run "$ENCODER" "$GPU_ID" &
        PIDS+=($!)
    done

    log ""
    log "Batch running: ${BATCH_ENCODERS[*]} (PIDs: ${PIDS[*]})"
    log "Waiting for batch to finish..."

    # Wait for each process and collect its exit code
    for j in "${!PIDS[@]}"; do
        PID="${PIDS[$j]}"
        ENCODER="${BATCH_ENCODERS[$j]}"
        set +e
        wait "$PID"
        EXIT_CODE=$?
        set -e
        collect_result "$ENCODER" "$EXIT_CODE"
    done

    log "Batch done."
    log ""
done

log "========================================================"
log "Sweep complete. Summary:"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log "Individual logs: $SWEEP_LOG_DIR"
log "========================================================"
