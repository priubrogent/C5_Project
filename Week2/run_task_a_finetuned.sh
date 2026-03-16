#!/bin/bash
# Run task_a_finetuned.py experiments in 3 parallel processes.
#
# Usage:
#   # Pretrained baseline:
#   bash run_task_a_finetuned.sh
#
#   # Fine-tuned PEFT adapter:
#   bash run_task_a_finetuned.sh --peft_adapter outputs/task_e_lora/lora_r8_focal_dice_noaug_sam-vit-base/best_adapter
#
#   # Merged checkpoint:
#   bash run_task_a_finetuned.sh --checkpoint outputs/task_e_lora/lora_r8_focal_dice_noaug_sam-vit-base/best_model_merged.pt
#
# All extra args (--output_dir, --split, --max_images, etc.) are forwarded to
# every worker process.
#
# Experiments split across 3 workers (single GPU, shared):
#   Worker 1: bbox_center  mask_centroid  random_mask_n1  random_mask_n3  random_mask_n5
#   Worker 2: random_bbox_n1  random_bbox_n3  random_bbox_n5  sift_best
#   Worker 3: sift_topk_n1  sift_topk_n3  sift_topk_n5  gt_bbox

set -e

SCRIPT="python task_a_finetuned.py"
LOG_DIR="logs_task_a_finetuned"
mkdir -p "$LOG_DIR"

# Timestamp for log files
TS=$(date +%Y%m%d_%H%M%S)

echo "=== run_task_a_finetuned.sh  [$(date)] ==="
echo "Args forwarded to workers: $*"
echo ""

# Worker 1: point strategies, single-point heavy
echo "[Worker 1] Starting: bbox_center mask_centroid random_mask_n1/3/5"
$SCRIPT "$@" \
    --experiments bbox_center mask_centroid random_mask_n1 random_mask_n3 random_mask_n5 \
    > "$LOG_DIR/worker1_${TS}.log" 2>&1 &
PID1=$!

# Worker 2: random_bbox + sift_best
echo "[Worker 2] Starting: random_bbox_n1/3/5 sift_best"
$SCRIPT "$@" \
    --experiments random_bbox_n1 random_bbox_n3 random_bbox_n5 sift_best \
    > "$LOG_DIR/worker2_${TS}.log" 2>&1 &
PID2=$!

# Worker 3: sift_topk + gt_bbox
echo "[Worker 3] Starting: sift_topk_n1/3/5 gt_bbox"
$SCRIPT "$@" \
    --experiments sift_topk_n1 sift_topk_n3 sift_topk_n5 gt_bbox \
    > "$LOG_DIR/worker3_${TS}.log" 2>&1 &
PID3=$!

echo ""
echo "PIDs: worker1=$PID1  worker2=$PID2  worker3=$PID3"
echo "Logs: $LOG_DIR/worker{1,2,3}_${TS}.log"
echo ""

# Wait and collect exit codes
FAIL=0

wait $PID1 && echo "[Worker 1] Done" || { echo "[Worker 1] FAILED (exit $?)"; FAIL=1; }
wait $PID2 && echo "[Worker 2] Done" || { echo "[Worker 2] FAILED (exit $?)"; FAIL=1; }
wait $PID3 && echo "[Worker 3] Done" || { echo "[Worker 3] FAILED (exit $?)"; FAIL=1; }

echo ""
if [ $FAIL -ne 0 ]; then
    echo "One or more workers failed. Check logs in $LOG_DIR/"
    exit 1
fi

echo "All workers finished successfully."
echo ""

# Print summary from the summary.json written by each worker
# (each worker writes its own partial summary.json — find the output_dir)
# Extract --output_dir from args if provided, else use the default
OUTPUT_DIR="outputs/task_a_finetuned"
for arg in "$@"; do
    shift
    if [ "$arg" = "--output_dir" ]; then
        OUTPUT_DIR="$1"
    fi
done

echo "=== Partial summaries written by each worker ==="
echo "(Full per-experiment metrics.json files are under $OUTPUT_DIR/<model_label>/)"
