#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/other/nrk_acid_feedback_experiment_13APR26/channel 1 A"
IMAGE_DIR="${GLOBAL_DIR}/frames"
MASK_DIR="${GLOBAL_DIR}/masks"
SAVE_PATH="${GLOBAL_DIR}/analysis"

SCRIPT2="SCRIPTS/core_pipeline/trajectories.py"

# -----------------------------
# TRAJECTORY PARAMETERS (determine using preprocess_gui.py or the WEBGUI)
# -----------------------------
MAX_DISTANCE=10
GRACE_PERIOD=3
RADIUS=0
RADIUS_Y=0
RADIUS_X=0
SHIFT_FRAME=45
SHIFT_XY="0 0"
SAVE_INTERVAL=500

# -----------------------------
# Ensure script exists
# -----------------------------
if [[ ! -f "$SCRIPT2" ]]; then
    echo "Script $SCRIPT2 not found."
    exit 1
fi

echo "--- Accessing ${GLOBAL_DIR} ---"
# -----------------------------
# Run trajectory processing
# -----------------------------
echo "Starting trajectory processing..."
python3 "$SCRIPT2" \
    --mask_dir "$MASK_DIR" \
    --image_dir "$IMAGE_DIR" \
    --save_path "$SAVE_PATH" \
    --max_distance "$MAX_DISTANCE" \
    --grace_period "$GRACE_PERIOD" \
    --radius "$RADIUS" \
    --radius_y "$RADIUS_Y" \
    --radius_x "$RADIUS_X" \
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy $SHIFT_XY \
    --save_interval "$SAVE_INTERVAL"