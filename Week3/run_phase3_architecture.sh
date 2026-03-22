#!/usr/bin/env bash
# ============================================================
# run_phase3_architecture.sh  —  Phase 3: Decoder Architecture
#
# 6 runs in parallel, one per GPU (GPUs 1,2,4,5,6,7):
#
#   gru_2l_512    GRU  2 layers  512 dim  dropout=0.3  (depth only)
#   gru_1l_1024   GRU  1 layer  1024 dim  dropout=0.0  (width only)
#   gru_2l_1024   GRU  2 layers 1024 dim  dropout=0.3  (depth + width)
#   lstm_2l_512   LSTM 2 layers  512 dim  dropout=0.3  (depth only)
#   lstm_1l_1024  LSTM 1 layer  1024 dim  dropout=0.0  (width only)
#   lstm_2l_1024  LSTM 2 layers 1024 dim  dropout=0.3  (depth + width)
#
# Baseline gru_1l_512 already exists as resnet50_gru_subword_tf_scheduled
# — compare against it, no need to re-run.
#
# NOTE: 1024-dim runs use batch_size=32 to avoid OOM.
#
# Fixed: resnet50 + subword + AdamW lr=1e-3 + weight_decay=1e-4
#
# Usage:
#   bash run_phase3_architecture.sh [OPTIONS]
#
# Options:
#   --data_root DIR
#   --out_root  DIR
#   --wandb_entity NAME
#   --wandb_project NAME
#   --epochs N            (default: 50)
#   --batch_size N        (default: 64, overridden to 32 for 1024-dim runs)
#   --gpu_ids IDS         (default: 1,2,4,5,6,7)
#   --tf_mode scheduled|on  (default: scheduled)
#   --no_wandb
# ============================================================
set -euo pipefail

# ---------- defaults ----------
DATA_ROOT="/home/msiau/data/tmp/amarcos/vizwiz_dataset"
OUT_ROOT="outputs"
WANDB_ENTITY="just-an-arbitrary-team-name"
WANDB_PROJECT="mcv-c5-image_captioning"
EPOCHS=50
BATCH_SIZE=64
GPU_IDS="1,2,4,5,6,7"
USE_WANDB=true
WANDB_SLEEP=10

# ---------- fixed backbone ----------
ENCODER="resnet50"
TEXT_REPR="subword"
OPTIMIZER="adamw"
LR="1e-3"
WEIGHT_DECAY="1e-4"
TF_MODE="scheduled"   # "scheduled" or "on"

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
        --tf_mode)       TF_MODE="$2";       shift 2 ;;
        --no_wandb)      USE_WANDB=false;    shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$TF_MODE" != "scheduled" && "$TF_MODE" != "on" ]]; then
    echo "Error: --tf_mode must be 'scheduled' or 'on'"; exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_IDS"

# ---------- configs: "suffix:decoder:layers:hidden_dim:embed_dim:dropout" ----------
# 1024-dim runs get batch_size halved automatically in launch_run
CONFIGS=(
    "gru_2l_512:gru:2:512:512:0.3"
    "gru_1l_1024:gru:1:1024:1024:0.0"
    "gru_2l_1024:gru:2:1024:1024:0.3"
    "lstm_2l_512:lstm:2:512:512:0.3"
    "lstm_1l_1024:lstm:1:1024:1024:0.0"
    "lstm_2l_1024:lstm:2:1024:1024:0.3"
)

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/phase3_architecture_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/phase3_summary.csv"

echo "run,decoder,layers,hidden_dim,dropout,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Phase 3 — Decoder Architecture sweep"
log "  backbone : $ENCODER | ${TEXT_REPR}"
log "  tf       : $TF_MODE"
log "  optimizer: $OPTIMIZER  lr=$LR  wd=$WEIGHT_DECAY"
log "  epochs   : $EPOCHS"
log "  GPUs     : ${GPUS[*]}"
log "  wandb    : $USE_WANDB"
log "  baseline : resnet50_gru_subword_tf_scheduled (gru 1l 512, already run)"
log "========================================================"

launch_run() {
    local SUFFIX="$1"
    local DECODER="$2"
    local LAYERS="$3"
    local HIDDEN_DIM="$4"
    local EMBED_DIM="$5"
    local DROPOUT="$6"
    local GPU_ID="$7"
    local RUN_NAME="${ENCODER}_${DECODER}_subword_${SUFFIX}_tf_${TF_MODE}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    # halve batch size for wide models to avoid OOM
    local BS="$BATCH_SIZE"
    if (( HIDDEN_DIM >= 1024 )); then
        BS=$(( BATCH_SIZE / 2 ))
    fi

    log "  Launching $RUN_NAME (decoder=$DECODER layers=$LAYERS dim=$HIDDEN_DIM dropout=$DROPOUT bs=$BS tf=$TF_MODE) on GPU $GPU_ID"

    local WANDB_ARGS=()
    if $USE_WANDB; then
        WANDB_ARGS=(--wandb --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT")
    fi

    local TF_ARGS=()
    if [[ "$TF_MODE" == "scheduled" ]]; then
        TF_ARGS=(--scheduled_tf)
    else
        TF_ARGS=(--teacher_forcing)
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --encoder        "$ENCODER" \
        --decoder        "$DECODER" \
        --decoder_layers "$LAYERS" \
        --hidden_dim     "$HIDDEN_DIM" \
        --embed_dim      "$EMBED_DIM" \
        --dropout        "$DROPOUT" \
        --text_repr      "$TEXT_REPR" \
        --epochs         "$EPOCHS" \
        --batch_size     "$BS" \
        --lr             "$LR" \
        --weight_decay   "$WEIGHT_DECAY" \
        --optimizer      "$OPTIMIZER" \
        "${TF_ARGS[@]}" \
        --lr_decay       0.5 \
        --lr_patience    5 \
        --es_patience    20 \
        --es_metric      val_loss \
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
    local SUFFIX="$1"
    local DECODER="$2"
    local LAYERS="$3"
    local HIDDEN_DIM="$4"
    local DROPOUT="$5"
    local EXIT_CODE="$6"
    local RUN_NAME="${ENCODER}_${DECODER}_subword_${SUFFIX}_tf_${TF_MODE}"
    local RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${RUN_NAME},${DECODER},${LAYERS},${HIDDEN_DIM},${DROPOUT},${ROW}" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> OK  |  $ROW"
    else
        echo "${RUN_NAME},${DECODER},${LAYERS},${HIDDEN_DIM},${DROPOUT},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/${RUN_NAME}.log"
    fi
}

PIDS=()
SUFFIXES=()
DECODERS=()
LAYERSS=()
HIDDEN_DIMS=()
DROPOUTS=()

for (( i=0; i<${#CONFIGS[@]}; i++ )); do
    IFS=':' read -r SUFFIX DECODER LAYERS HIDDEN_DIM EMBED_DIM DROPOUT <<< "${CONFIGS[$i]}"
    GPU_ID="${GPUS[$i]}"
    SUFFIXES+=("$SUFFIX")
    DECODERS+=("$DECODER")
    LAYERSS+=("$LAYERS")
    HIDDEN_DIMS+=("$HIDDEN_DIM")
    DROPOUTS+=("$DROPOUT")

    launch_run "$SUFFIX" "$DECODER" "$LAYERS" "$HIDDEN_DIM" "$EMBED_DIM" "$DROPOUT" "$GPU_ID" &
    PIDS+=($!)

    if (( i < ${#CONFIGS[@]}-1 )); then
        log "  Sleeping ${WANDB_SLEEP}s before next launch..."
        sleep "$WANDB_SLEEP"
    fi
done

log ""
log "All runs launched (PIDs: ${PIDS[*]}). Waiting..."

for j in "${!PIDS[@]}"; do
    set +e
    wait "${PIDS[$j]}"
    EXIT_CODE=$?
    set -e
    collect_result "${SUFFIXES[$j]}" "${DECODERS[$j]}" "${LAYERSS[$j]}" "${HIDDEN_DIMS[$j]}" "${DROPOUTS[$j]}" "$EXIT_CODE"
done

log ""
log "========================================================"
log "Phase 3 complete. Summary (include baseline for comparison):"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log ""
log "Baseline (gru 1l 512): resnet50_gru_subword_tf_scheduled"
log "  bleu1=67.95  bleu2=46.92  rougeL=29.68  meteor=38.76  test_loss=3.2902"
log "  (Phase 2 winner adamw_1e3: bleu1=65.66 bleu2=45.20 meteor=41.79 test_loss=2.9934)"
log ""
log "To find the winner:"
log "  python - <<'EOF'"
log "  import json, glob"
log "  paths = glob.glob('${OUT_ROOT}/resnet50_*_subword_*/history.json')"
log "  for p in sorted(paths):"
log "      h = json.load(open(p)); best = min(h, key=lambda r: r['val_loss'])"
log "      print(p, '-> val_loss:', round(best['val_loss'],4), 'epoch:', best['epoch'])"
log "  EOF"
log "========================================================"
