#!/usr/bin/env bash
# ============================================================
# run_phase1_remaining.sh  —  Phase 1: word + subword TF sweep
#
# Runs the 6 remaining Phase 1 experiments (char already running):
#   word     × {tf_on, tf_off, tf_scheduled}
#   subword  × {tf_on, tf_off, tf_scheduled}
#
# GPU assignment (1 run per GPU, all 6 in parallel):
#   GPU 0: word_tf_on
#   GPU 1: word_tf_off
#   GPU 2: word_tf_scheduled
#   GPU 3: subword_tf_on
#   GPU 4: subword_tf_off
#   GPU 5: subword_tf_scheduled
#
# Fixed: vgg16 + gru + 1 layer + 512 dim + Adam lr=1e-3
#
# Usage:
#   bash run_phase1_remaining.sh [OPTIONS]
#
# Options:
#   --data_root DIR
#   --out_root  DIR
#   --wandb_entity NAME
#   --wandb_project NAME
#   --epochs N            (default: 50)
#   --batch_size N        (default: 64)
#   --gpu_ids IDS         (default: 0,1,2,4,5,6)
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
GPU_IDS="0,1,2,4,5,6"
USE_WANDB=true
WANDB_SLEEP=10

# ---------- fixed backbone ----------
ENCODER="resnet50"
DECODER="gru"
DECODER_LAYERS=1
HIDDEN_DIM=512
EMBED_DIM=512

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

if [[ ${#GPUS[@]} -lt 6 ]]; then
    echo "ERROR: need at least 6 GPUs (got ${#GPUS[@]}). Provide --gpu_ids 0,1,2,4,5,6."
    exit 1
fi

SWEEP_LOG_DIR="${OUT_ROOT}/sweep_logs"
mkdir -p "$SWEEP_LOG_DIR"

MASTER_LOG="${SWEEP_LOG_DIR}/phase1_remaining_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_CSV="${SWEEP_LOG_DIR}/phase1_remaining_summary.csv"

echo "run,tokenizer,tf_mode,test_loss,bleu1,bleu2,rougeL,meteor,status" > "$SUMMARY_CSV"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

log "========================================================"
log "Phase 1 remaining — word + subword TF sweep"
log "  backbone : $ENCODER + $DECODER x${DECODER_LAYERS} | dim=${HIDDEN_DIM}"
log "  epochs   : $EPOCHS  batch=$BATCH_SIZE"
log "  GPUs     : ${GPUS[*]}"
log "  wandb    : $USE_WANDB"
log "========================================================"

# ---------- per-tokenizer config ----------
# "text_repr:max_len:run_label"
declare -A MAX_LEN=( [word]=35 [subword]=50 )

run_single() {
    local TEXT_REPR="$1"
    local TF_FLAG="$2"
    local GPU_ID="$3"
    local ML="${MAX_LEN[$TEXT_REPR]}"
    local RUN_NAME="${ENCODER}_${DECODER}_${TEXT_REPR}_tf_${TF_FLAG}"
    local RUN_LOG="${SWEEP_LOG_DIR}/${RUN_NAME}.log"

    log "  Launching $RUN_NAME on GPU $GPU_ID"

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
        --max_len        "$ML" \
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
        >> "$RUN_LOG" 2>&1
}

collect_result() {
    local TEXT_REPR="$1"
    local TF_FLAG="$2"
    local EXIT_CODE="$3"
    local RUN_NAME="${ENCODER}_${DECODER}_${TEXT_REPR}_tf_${TF_FLAG}"
    local RESULTS_FILE="${OUT_ROOT}/${RUN_NAME}/test_results.json"

    if [[ $EXIT_CODE -eq 0 && -f "$RESULTS_FILE" ]]; then
        ROW=$(python - <<EOF
import json
d = json.load(open("$RESULTS_FILE"))
print(f"{d.get('test_loss',0):.4f},{d.get('bleu1',0):.2f},{d.get('bleu2',0):.2f},{d.get('rougeL',0):.2f},{d.get('meteor',0):.2f},ok")
EOF
)
        echo "${RUN_NAME},${TEXT_REPR},${TF_FLAG},${ROW}" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> OK  |  $ROW"
    else
        echo "${RUN_NAME},${TEXT_REPR},${TF_FLAG},,,,,FAILED" >> "$SUMMARY_CSV"
        log "  $RUN_NAME -> FAILED (exit=$EXIT_CODE). See ${SWEEP_LOG_DIR}/${RUN_NAME}.log"
    fi
}

# 6 configs, one per GPU
# Format: "text_repr:tf_flag"
CONFIGS=(
    "word:on"
    "word:off"
    "word:scheduled"
    "subword:on"
    "subword:off"
    "subword:scheduled"
)

PIDS=()
TEXT_REPRS=()
TF_FLAGS_LIST=()

for (( i=0; i<${#CONFIGS[@]}; i++ )); do
    IFS=':' read -r TEXT_REPR TF_FLAG <<< "${CONFIGS[$i]}"
    GPU_ID="${GPUS[$i]}"
    TEXT_REPRS+=("$TEXT_REPR")
    TF_FLAGS_LIST+=("$TF_FLAG")

    run_single "$TEXT_REPR" "$TF_FLAG" "$GPU_ID" &
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
    collect_result "${TEXT_REPRS[$j]}" "${TF_FLAGS_LIST[$j]}" "$EXIT_CODE"
done

log ""
log "========================================================"
log "Phase 1 remaining complete. Summary:"
column -t -s ',' "$SUMMARY_CSV" | tee -a "$MASTER_LOG"
log ""
log "To compare all 9 Phase 1 runs (including char from run_phase1_tf.sh):"
log "  python - <<'EOF'"
log "  import json, glob"
log "  rows = []"
log "  for p in glob.glob('${OUT_ROOT}/vgg16_gru_*/history.json'):"
log "      h = json.load(open(p)); best = min(h, key=lambda r: r['val_loss'])"
log "      rows.append((best['val_loss'], p, best['epoch']))"
log "  for v, p, e in sorted(rows): print(f'{v:.4f}  ep={e}  {p}')"
log "  EOF"
log "========================================================"
