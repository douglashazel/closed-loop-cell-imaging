#!/bin/bash
# Stage 1, tracking only: trajectories and fluorescence from the masks already
# in masks/ for one experiment.
#
#   bash run_trajectories.sh <experiment_dir | config.yaml>
#
# Parameters come from the tracking section of
# <experiment_dir>/pipeline_config.yaml (see run_processes.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_trajectories.sh <experiment_dir | config.yaml>" >&2
    exit 2
fi

# -----------------------------
# Parameters
# -----------------------------
PARAMS=$(python3 "$ROOT/SCRIPTS/core_pipeline/pipeline_config.py" prepare "$1" run_trajectories.sh tracking)
eval "$PARAMS"
cd "$ROOT"

echo "--- Accessing ${EXP_DIR} ---"
echo "Parameters from ${CONFIG_FILE}; saved to ${RUN_RECORD}"

# -----------------------------
# Run trajectory processing
# -----------------------------
echo ">>> STAGE: TRAJECTORIES <<<"
python3 -u SCRIPTS/core_pipeline/trajectories.py \
    --mask_dir "$MASK_DIR" \
    --image_dir "$IMAGE_DIR" \
    --save_path "$ANALYSIS_DIR" \
    --max_distance "$MAX_DISTANCE" \
    --grace_period "$GRACE_PERIOD" \
    --radius "$RADIUS" \
    --radius_y "$RADIUS_Y" \
    --radius_x "$RADIUS_X" \
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy "$SHIFT_X" "$SHIFT_Y" \
    --save_interval "$SAVE_INTERVAL" \
    --workers "$WORKERS"

echo ">>> DONE <<<"
