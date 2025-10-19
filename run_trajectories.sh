#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="c2c12_thapsigargin_2/frames"
MASK_DIR="c2c12_thapsigargin_2/masks"
SAVE_PATH="c2c12_thapsigargin_2/analysis"

SCRIPT2="trajectories.py"

SHIFT_FRAME=5
SHIFT_DX=1.7487
SHIFT_DY=-0.5875

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
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy "$SHIFT_DX" "$SHIFT_DY"