#!/usr/bin/env bash
# ============================================================
# run_tokenizer_sweep_parallel.sh
# Sweeps tokenization strategies with fixed encoder and decoder.
# All 3 experiments run simultaneously, one per GPU.
#
# Configurations:
#   char     char-level,    max_len=150, vocab=80
#   word     word-level,    max_len=35,  vocab=~corpus-dependent
#   subword  BPE subword,   max_len=50,  vocab=4000
#
# Usage:
#   bash run_tokenizer_sweep_parallel.sh [OPTIONS]
#
# Options:
#   --encoder NAME        Fixed encoder          (default: vgg16)
#   --decoder NAME        Fixed decoder          (default: gru)
#   --data_root DIR       Path to vizwiz_dataset
#   --out_root  DIR       Path to output root
#   --wandb_entity NAME   W&B entity / username
#   --wandb_project NAME  W&B project name
#   --epochs N            Epochs per run         (default: 50)
#   --batch_size N        Batch size             (default: 64)
#   --gpu_ids IDS         Comma-separated GPU ids (default: 0,1,2)
#   --no_wandb            Disable W&B logging
# ============================================================
set -euo pipefail

# ---------- defaults ----------
ENCODER="vgg16"
DECODER="gru"
DATA_ROOT="/home/msiau/data/tmp/amarcos/vizwiz_dataset"
OUT_ROOT="outputs"
WANDB_ENTITY="just-an-arbitrary-team-name"
WANDB_PROJECT="mcv-c5-image_captioning"
EPOCHS=50
BATCH_SIZE=64
GPU_IDS="0,1,2"
USE_WANDB=true
WANDB_SLEEP=10   # seconds between launches to avoid W&B rate limits

# ---------- tokenizer configs ----------
# Each entry: "text_repr:max_len"
CONFIGS=(
    "char:150"
    "word:35"
    "subword:50"
)

# ---------- parse CLI ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder)       ENCODER="$2";       shift 2 ;;
        --decoder)       DECODER="$2";       shift 2 ;;
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

IFS=',' read -ra GPUS <<< "$GPU_IDS"

if [[ ${#GPUS[@]} -lt ${#CONFIGS[@]} ]]; then
    echo "WARNING: fewer GPUs (${#GPUS[@]}) than configs (${#CONFIGS[@]}). Some runs will be skipped."
fi

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/tokenizer_sweep_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/tokenizer_summary.csv"

echo "encoder,decoder,text_repr,max_len,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Tokenizer sweep started"
log "  encoder     : $ENCODER"
log "  decoder     : $DECODER"
log "  configs     : ${CONFIGS[*]}"
log "  epochs      : $EPOCHS"
log "  batch_size  : $BATCH_SIZE"
log "  GPUs        : ${GPUS[*]}"
log "  data_root   : $DATA_ROOT"
log "  out_root    : $OUT_ROOT"
log "  wandb       : $USE_WANDB (entity=$WANDB_ENTITY, project=$WANDB_PROJECT)"
log "========================================================"

launch_run() {
    local TEXT_REPR="$1"
    local MAX_LEN="$2"
    local GPU_ID="$3"
    local RUN_NAME="${ENCODER}_${DECODER}_${TEXT_REPR}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log "  Launching $RUN_NAME on GPU $GPU_ID -> $RUN_LOG"

    local WANDB_ARGS=()
    if $USE_WANDB; then
        WANDB_ARGS=(--wandb --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --encoder        "$ENCODER" \
        --decoder        "$DECODER" \
        --decoder_layers 1 \
        --hidden_dim     512 \
        --embed_dim      512 \
        --dropout        0.0 \
        --text_repr      "$TEXT_REPR" \
        --max_len        "$MAX_LEN" \
        --epochs         "$EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --lr             1e-3 \
        --weight_decay   1e-4 \
        --optimizer      adam \
        --teacher_forcing \
        --lr_decay       0.5 \
        --lr_patience    5 \
        --es_patience    10 \
        --es_metric      meteor \
        --run_name       "$RUN_NAME" \
        --seed           42 \
        --num_workers    4 \
        --data_root      "$DATA_ROOT" \
        --out_root       "$OUT_ROOT" \
        --device         cuda \
        "${WANDB_ARGS[@]}" \
        > "$RUN_LOG" 2>&1
}

collect_result() {
    local TEXT_REPR="$1"
    local MAX_LEN="$2"
    local EXIT_CODE="$3"
    local RUN_NAME="${ENCODER}_${DECODER}_${TEXT_REPR}"
    local RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${ENCODER},${DECODER},${TEXT_REPR},${MAX_LEN},${ROW}" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> OK  |  $ROW"
    else
        echo "${ENCODER},${DECODER},${TEXT_REPR},${MAX_LEN},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/${RUN_NAME}.log"
    fi
}

# All 3 configs fit on 3 GPUs — launch everything at once
PIDS=()
TEXT_REPRS=()
MAX_LENS=()

for (( i=0; i<${#CONFIGS[@]} && i<${#GPUS[@]}; i++ )); do
    IFS=':' read -r TEXT_REPR MAX_LEN <<< "${CONFIGS[$i]}"
    GPU_ID="${GPUS[$i]}"
    TEXT_REPRS+=("$TEXT_REPR")
    MAX_LENS+=("$MAX_LEN")

    launch_run "$TEXT_REPR" "$MAX_LEN" "$GPU_ID" &
    PIDS+=($!)

    # Sleep between launches to avoid W&B rate-limit errors
    if (( i < ${#CONFIGS[@]}-1 )); then
        log "  Sleeping ${WANDB_SLEEP}s before next launch..."
        sleep "$WANDB_SLEEP"
    fi
done

log ""
log "All runs launched (PIDs: ${PIDS[*]}). Waiting..."
log ""

for j in "${!PIDS[@]}"; do
    set +e
    wait "${PIDS[$j]}"
    EXIT_CODE=$?
    set -e
    collect_result "${TEXT_REPRS[$j]}" "${MAX_LENS[$j]}" "$EXIT_CODE"
done

log ""
log "========================================================"
log "Tokenizer sweep complete. Summary:"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log "Individual logs: $SWEEP_LOG_DIR"
log "========================================================"
