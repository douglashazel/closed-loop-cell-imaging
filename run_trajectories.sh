#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
GLOBAL_DIR="DMSO_TEST"
IMAGE_DIR="${GLOBAL_DIR}/frames"
MASK_DIR="${GLOBAL_DIR}/masks"
SAVE_PATH="${GLOBAL_DIR}/analysis"

SCRIPT2="trajectories_optimized.py"

SHIFT_FRAME=46
SHIFT_DX=0
SHIFT_DY=0

# -----------------------------
# Ensure script exists
# -----------------------------
if [[ ! -f "$SCRIPT2" ]]; then
    echo "Script $SCRIPT2 not found."
    exit 1
fi

# -----------------------------
# Run trajectory processing
# -----------------------------
echo "Starting trajectory processing..."
python3 "$SCRIPT2" \
    --mask_dir "$MASK_DIR" \
    --image_dir "$IMAGE_DIR" \
    --save_path "$SAVE_PATH" \
    --max_distance 20.0 \
    --grace_period 3 \
    --radius 0 \
    --y_shift 0 \
    --x_shift 0 \
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy "$SHIFT_DX" "$SHIFT_DY" \
    --save_interval 500