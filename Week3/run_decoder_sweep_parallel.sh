#!/usr/bin/env bash
# ============================================================
# run_decoder_sweep_parallel.sh
# Sweeps decoder configurations with a fixed encoder.
# Runs up to 3 experiments in parallel across 3 GPUs.
#
# Configurations:
#   gru_1l   GRU,  1 layer, 512 hidden, dropout=0.0
#   lstm_1l  LSTM, 1 layer, 512 hidden, dropout=0.0
#   gru_2l   GRU,  2 layers, 512 hidden, dropout=0.3
#   lstm_2l  LSTM, 2 layers, 512 hidden, dropout=0.3
#
# Usage:
#   bash run_decoder_sweep_parallel.sh [OPTIONS]
#
# Options:
#   --encoder NAME        Encoder to fix for all runs  !! SET THIS !!
#   --data_root DIR       Path to vizwiz_dataset
#   --out_root  DIR       Path to output root
#   --wandb_entity NAME   W&B entity / username
#   --wandb_project NAME  W&B project name
#   --epochs N            Epochs per run            (default: 50)
#   --batch_size N        Batch size                (default: 64)
#   --gpu_ids IDS         Comma-separated GPU ids   (default: 0,1,2)
#   --no_wandb            Disable W&B logging
# ============================================================
set -euo pipefail

# ---------- defaults ----------
ENCODER="vgg16"
DATA_ROOT="/home/msiau/data/tmp/amarcos/vizwiz_dataset"
OUT_ROOT="outputs"
WANDB_ENTITY="just-an-arbitrary-team-name"
WANDB_PROJECT="mcv-c5-image_captioning"
EPOCHS=50
BATCH_SIZE=64
GPU_IDS="0,1,2"
USE_WANDB=true
WANDB_SLEEP=10   # seconds to sleep between run launches to avoid W&B rate limits

# ---------- parse CLI ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder)       ENCODER="$2";       shift 2 ;;
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
NUM_GPUS=${#GPUS[@]}

# ---------- decoder configurations ----------
# Each entry: "run_suffix:decoder:layers:dropout"
CONFIGS=(
    "gru_1l:gru:1:0.0"
    "lstm_1l:lstm:1:0.0"
    "gru_2l:gru:2:0.3"
    "lstm_2l:lstm:2:0.3"
)

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/decoder_sweep_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/decoder_summary.csv"

echo "encoder,decoder,layers,dropout,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Decoder sweep started"
log "  encoder     : $ENCODER"
log "  configs     : ${CONFIGS[*]}"
log "  epochs      : $EPOCHS"
log "  batch_size  : $BATCH_SIZE"
log "  GPUs        : ${GPUS[*]}"
log "  data_root   : $DATA_ROOT"
log "  out_root    : $OUT_ROOT"
log "  wandb       : $USE_WANDB (entity=$WANDB_ENTITY, project=$WANDB_PROJECT)"
log "========================================================"

launch_run() {
    local RUN_SUFFIX="$1"
    local DECODER="$2"
    local LAYERS="$3"
    local DROPOUT="$4"
    local GPU_ID="$5"
    local RUN_NAME="${ENCODER}_${RUN_SUFFIX}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log "  Launching $RUN_NAME on GPU $GPU_ID -> $RUN_LOG"

    local WANDB_ARGS=()
    if $USE_WANDB; then
        WANDB_ARGS=(--wandb --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --encoder        "$ENCODER" \
        --decoder        "$DECODER" \
        --decoder_layers "$LAYERS" \
        --dropout        "$DROPOUT" \
        --hidden_dim     512 \
        --embed_dim      512 \
        --text_repr      char \
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
    local RUN_SUFFIX="$1"
    local DECODER="$2"
    local LAYERS="$3"
    local DROPOUT="$4"
    local EXIT_CODE="$5"
    local RUN_NAME="${ENCODER}_${RUN_SUFFIX}"
    local RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${ENCODER},${DECODER},${LAYERS},${DROPOUT},${ROW}" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> OK  |  $ROW"
    else
        echo "${ENCODER},${DECODER},${LAYERS},${DROPOUT},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/${RUN_NAME}.log"
    fi
}

# Run configs in batches of NUM_GPUS
i=0
while [[ $i -lt ${#CONFIGS[@]} ]]; do
    PIDS=()
    BATCH_SUFFIXES=()
    BATCH_DECODERS=()
    BATCH_LAYERS=()
    BATCH_DROPOUTS=()

    for (( g=0; g<NUM_GPUS && i<${#CONFIGS[@]}; g++, i++ )); do
        IFS=':' read -r SUFFIX DECODER LAYERS DROPOUT <<< "${CONFIGS[$i]}"
        GPU_ID="${GPUS[$g]}"
        BATCH_SUFFIXES+=("$SUFFIX")
        BATCH_DECODERS+=("$DECODER")
        BATCH_LAYERS+=("$LAYERS")
        BATCH_DROPOUTS+=("$DROPOUT")

        launch_run "$SUFFIX" "$DECODER" "$LAYERS" "$DROPOUT" "$GPU_ID" &
        PIDS+=($!)

        # Sleep between launches to avoid W&B rate-limit errors
        if (( g < NUM_GPUS-1 && i+1 < ${#CONFIGS[@]} )); then
            log "  Sleeping ${WANDB_SLEEP}s before next launch..."
            sleep "$WANDB_SLEEP"
        fi
    done

    log ""
    log "Batch running: ${BATCH_SUFFIXES[*]} (PIDs: ${PIDS[*]})"
    log "Waiting for batch to finish..."

    for j in "${!PIDS[@]}"; do
        set +e
        wait "${PIDS[$j]}"
        EXIT_CODE=$?
        set -e
        collect_result "${BATCH_SUFFIXES[$j]}" "${BATCH_DECODERS[$j]}" \
                       "${BATCH_LAYERS[$j]}" "${BATCH_DROPOUTS[$j]}" "$EXIT_CODE"
    done

    log "Batch done."
    log ""

    # Sleep before starting the next batch too
    if [[ $i -lt ${#CONFIGS[@]} ]]; then
        log "Sleeping ${WANDB_SLEEP}s before next batch..."
        sleep "$WANDB_SLEEP"
    fi
done

log "========================================================"
log "Decoder sweep complete. Summary:"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log "Individual logs: $SWEEP_LOG_DIR"
log "========================================================"
