#!/bin/bash
# Stage 1 for one experiment: Cellpose segmentation and tracking side by side,
# then the pre-analysis plots.
#
#   bash run_processes.sh [--skip-segmentation] <experiment_dir | config.yaml>
#
# Parameters come from <experiment_dir>/pipeline_config.yaml (sections
# segmentation and tracking), written by TUNE_GUI or copied from
# configs/example_c2c12_chamber_A.yaml, so this script needs no edits. The
# values used are saved under <experiment_dir>/analysis/run_history/.
# --skip-segmentation tracks the masks already in masks/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LABEL="run_processes.sh"
SECTIONS="segmentation tracking"
if [[ "${1:-}" == "--skip-segmentation" ]]; then
    RUN_LABEL="run_processes.sh --skip-segmentation"
    SECTIONS="tracking"
    shift
fi
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_processes.sh [--skip-segmentation] <experiment_dir | config.yaml>" >&2
    exit 2
fi

# -----------------------------
# Parameters (stops here if the config is missing or invalid)
# -----------------------------
PARAMS=$(python3 "$ROOT/SCRIPTS/core_pipeline/pipeline_config.py" prepare "$1" "$RUN_LABEL" $SECTIONS)
eval "$PARAMS"
cd "$ROOT"

echo "--- Accessing ${EXP_DIR} ---"
echo "Parameters from ${CONFIG_FILE}; saved to ${RUN_RECORD}"

# -----------------------------
# Run segmentation
# -----------------------------
PID1=""
echo ">>> STAGE: SEGMENTATION <<<"
if [[ "$SECTIONS" == *segmentation* ]]; then
    python3 -u SCRIPTS/core_pipeline/segmentation.py \
        --image_dir "$IMAGE_DIR" \
        --mask_dir "$MASK_DIR" \
        --flow_threshold "$FLOW_THRESHOLD" \
        --cellprob_threshold "$CELLPROB_THRESHOLD" \
        --niter "$NITER" \
        --diameter "$DIAMETER" &
    PID1=$!
    sleep 5 # small buffer
else
    echo "Skipping segmentation; tracking the masks already in $MASK_DIR"
fi

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
    --workers "$WORKERS" &
PID2=$!

# `wait $PID1 $PID2` would only report the last status. If segmentation fails,
# stop tracking too (it would otherwise wait for masks that never come).
if [[ -n "$PID1" ]]; then
    wait "$PID1" || { echo "Segmentation failed."; kill "$PID2" 2>/dev/null; exit 1; }
fi
wait "$PID2"

# -----------------------------
# Pre-analysis plots
# -----------------------------
echo ">>> STAGE: PRE-ANALYSIS <<<"
python3 -u SCRIPTS/core_pipeline/PreAnalysis.py \
    --exp "$EXP_DIR" \
    --analysis_dir "$ANALYSIS_DIR"

echo ">>> DONE <<<"
