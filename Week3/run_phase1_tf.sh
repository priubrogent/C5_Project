#!/usr/bin/env bash
# ============================================================
# run_phase1_tf.sh  —  Phase 1: Teacher Forcing strategy
#
# 3 runs in parallel, one per GPU:
#   tf_on        teacher forcing always on   (tf_prob=1.0)
#   tf_off       teacher forcing always off  (tf_prob=0.0)
#   tf_scheduled linear decay 1.0→0.0 over all epochs
#
# Fixed: vgg16 + gru + 1 layer + 512 dim + char + Adam lr=1e-3
#
# Winner picked manually by val_loss from history.json.
# Pass winning config to Phase 2 via --encoder/--decoder/etc.
#
# Usage:
#   bash run_phase1_tf.sh [OPTIONS]
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
ENCODER="vgg16"
DECODER="gru"
DECODER_LAYERS=1
HIDDEN_DIM=512
EMBED_DIM=512
TEXT_REPR="char"

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

# ---------- configs: "run_suffix:tf_flag" ----------
# tf_flag: "on" | "off" | "scheduled"
CONFIGS=(
    "tf_on:on"
    "tf_off:off"
    "tf_scheduled:scheduled"
)

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/phase1_tf_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/phase1_summary.csv"

echo "run,tf_mode,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Phase 1 — Teacher Forcing sweep"
log "  backbone : $ENCODER + $DECODER x${DECODER_LAYERS} | ${TEXT_REPR} | dim=${HIDDEN_DIM}"
log "  epochs   : $EPOCHS  batch=$BATCH_SIZE"
log "  GPUs     : ${GPUS[*]}"
log "  wandb    : $USE_WANDB"
log "========================================================"

launch_run() {
    local SUFFIX="$1"
    local TF_FLAG="$2"
    local GPU_ID="$3"
    local RUN_NAME="${ENCODER}_${DECODER}_${SUFFIX}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log "  Launching $RUN_NAME (tf=$TF_FLAG) on GPU $GPU_ID"

    local TF_ARGS=()
    case "$TF_FLAG" in
        on)        TF_ARGS=(--teacher_forcing) ;;
        off)       TF_ARGS=(--no_teacher_forcing) ;;
        scheduled) TF_ARGS=(--scheduled_tf) ;;
    esac

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
        --epochs         "$EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --lr             1e-3 \
        --weight_decay   1e-4 \
        --optimizer      adam \
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
        "${TF_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        > "$RUN_LOG" 2>&1
}

collect_result() {
    local SUFFIX="$1"
    local TF_FLAG="$2"
    local EXIT_CODE="$3"
    local RUN_NAME="${ENCODER}_${DECODER}_${SUFFIX}"
    local RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${RUN_NAME},${TF_FLAG},${ROW}" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> OK  |  $ROW"
    else
        echo "${RUN_NAME},${TF_FLAG},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/${RUN_NAME}.log"
    fi
}

# All 3 fit on 3 GPUs — launch simultaneously
PIDS=()
SUFFIXES=()
TF_FLAGS=()

for (( i=0; i<${#CONFIGS[@]}; i++ )); do
    IFS=':' read -r SUFFIX TF_FLAG <<< "${CONFIGS[$i]}"
    GPU_ID="${GPUS[$i]}"
    SUFFIXES+=("$SUFFIX")
    TF_FLAGS+=("$TF_FLAG")

    launch_run "$SUFFIX" "$TF_FLAG" "$GPU_ID" &
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
    collect_result "${SUFFIXES[$j]}" "${TF_FLAGS[$j]}" "$EXIT_CODE"
done

log ""
log "========================================================"
log "Phase 1 complete. Summary (pick winner by best val_loss):"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log ""
log "To find the winner:"
log "  python - <<'EOF'"
log "  import json, glob"
log "  for p in glob.glob('${OUT_ROOT}/vgg16_gru_tf_*/history.json'):"
log "      h = json.load(open(p)); best = min(h, key=lambda r: r['val_loss'])"
log "      print(p, '-> best val_loss:', best['val_loss'], 'epoch:', best['epoch'])"
log "  EOF"
log "========================================================"
