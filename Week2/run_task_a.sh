#!/bin/bash
set -e

OUTPUT_BASE="./outputs/task_a_arnau"

run() {
    echo "=== $* ==="
    python task_a.py --output_dir "$OUTPUT_BASE" --qual_per_seq 10 "$@"
}

run --strategy bbox_center   --num_points 1

run --strategy mask_centroid --num_points 1

run --strategy random_mask   --num_points 1
run --strategy random_mask   --num_points 3
run --strategy random_mask   --num_points 5

run --strategy random_bbox   --num_points 1
run --strategy random_bbox   --num_points 3
run --strategy random_bbox   --num_points 5

run --strategy sift_best     --num_points 1

run --strategy sift_topk     --num_points 1
run --strategy sift_topk     --num_points 3
run --strategy sift_topk     --num_points 5


# python task_a.py --output_dir "./outputs/task_a_arnau" --strategy random_mask --num_points 3