#!/usr/bin/env bash
# ============================================================
# run_phase2_optimizer.sh  —  Phase 2: Optimizer + Learning Rate
#
# 3 runs in parallel, one per GPU:
#   adamw_1e3   AdamW  lr=1e-3  (Adam baseline but with correct weight decay)
#   adamw_3e4   AdamW  lr=3e-4  (lower LR variant)
#   sgd_1e2     SGD    lr=1e-2  (SGD needs ~10x higher LR than Adam)
#
# Baseline adam lr=1e-3 already exists as resnet50_gru_subword_tf_scheduled
# — compare against it, no need to re-run.
#
# Fixed: resnet50 + gru + 1 layer + 512 dim + subword (max_len=50)
#        + tf_scheduled + dropout=0.0
#
# Usage:
#   bash run_phase2_optimizer.sh [OPTIONS]
#
# Options:
#   --data_root DIR
#   --out_root  DIR
#   --wandb_entity NAME
#   --wandb_project NAME
#   --epochs N            (default: 50)
#   --batch_size N        (default: 64)
#   --gpu_ids IDS         (default: 0,1,2)
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
GPU_IDS="0,1,2"
USE_WANDB=true
WANDB_SLEEP=10

# ---------- fixed backbone ----------
ENCODER="resnet50"
DECODER="gru"
DECODER_LAYERS=1
HIDDEN_DIM=512
EMBED_DIM=512
TEXT_REPR="subword"
MAX_LEN=50

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

IFS=',' read -ra GPUS <<< "$GPU_IDS"

# ---------- configs: "run_suffix:optimizer:lr" ----------
CONFIGS=(
    "adamw_1e3:adamw:1e-3"
    "adamw_3e4:adamw:3e-4"
    "sgd_1e2:sgd:1e-2"
)

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/phase2_optimizer_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/phase2_summary.csv"

echo "run,optimizer,lr,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Phase 2 — Optimizer + LR sweep"
log "  backbone : $ENCODER + $DECODER x${DECODER_LAYERS} | ${TEXT_REPR} max_len=${MAX_LEN} | dim=${HIDDEN_DIM}"
log "  tf       : scheduled"
log "  epochs   : $EPOCHS  batch=$BATCH_SIZE"
log "  GPUs     : ${GPUS[*]}"
log "  wandb    : $USE_WANDB"
log "  baseline : resnet50_gru_subword_tf_scheduled (adam lr=1e-3, already run)"
log "========================================================"

launch_run() {
    local SUFFIX="$1"
    local OPT="$2"
    local LR="$3"
    local GPU_ID="$4"
    local RUN_NAME="${ENCODER}_${DECODER}_subword_${SUFFIX}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log "  Launching $RUN_NAME (opt=$OPT lr=$LR) on GPU $GPU_ID"

    local WANDB_ARGS=()
    if $USE_WANDB; then
        WANDB_ARGS=(--wandb --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT")
    fi

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
        --encoder        "$ENCODER" \
        --decoder        "$DECODER" \
        --decoder_layers "$DECODER_LAYERS" \
        --hidden_dim     "$HIDDEN_DIM" \
        --embed_dim      "$EMBED_DIM" \
        --dropout        0.0 \
        --text_repr      "$TEXT_REPR" \
        --max_len        "$MAX_LEN" \
        --epochs         "$EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --lr             "$LR" \
        --weight_decay   1e-4 \
        --optimizer      "$OPT" \
        --scheduled_tf \
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
    local OPT="$2"
    local LR="$3"
    local EXIT_CODE="$4"
    local RUN_NAME="${ENCODER}_${DECODER}_subword_${SUFFIX}"
    local RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${RUN_NAME},${OPT},${LR},${ROW}" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> OK  |  $ROW"
    else
        echo "${RUN_NAME},${OPT},${LR},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/${RUN_NAME}.log"
    fi
}

PIDS=()
SUFFIXES=()
OPTS=()
LRS=()

for (( i=0; i<${#CONFIGS[@]}; i++ )); do
    IFS=':' read -r SUFFIX OPT LR <<< "${CONFIGS[$i]}"
    GPU_ID="${GPUS[$i]}"
    SUFFIXES+=("$SUFFIX")
    OPTS+=("$OPT")
    LRS+=("$LR")

    launch_run "$SUFFIX" "$OPT" "$LR" "$GPU_ID" &
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
    collect_result "${SUFFIXES[$j]}" "${OPTS[$j]}" "${LRS[$j]}" "$EXIT_CODE"
done

log ""
log "========================================================"
log "Phase 2 complete. Summary (include baseline for comparison):"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log ""
log "Baseline (adam lr=1e-3): resnet50_gru_subword_tf_scheduled"
log "  bleu1=67.52  bleu2=46.45  rougeL=30.17  meteor=38.14  val_loss=3.2500"
log ""
log "To find the winner:"
log "  python - <<'EOF'"
log "  import json, glob"
log "  paths = glob.glob('${OUT_ROOT}/resnet50_gru_subword_*/history.json')"
log "  for p in paths:"
log "      h = json.load(open(p)); best = min(h, key=lambda r: r['val_loss'])"
log "      print(p, '-> val_loss:', best['val_loss'], 'epoch:', best['epoch'])"
log "  EOF"
log "========================================================"
