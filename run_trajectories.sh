#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="pc3_thapsigargin_2/frames"
MASK_DIR="pc3_thapsigargin_2/masks"
SAVE_PATH="pc3_thapsigargin_2/analysis"

SCRIPT2="trajectories.py"

SHIFT_FRAME=5
SHIFT_DX=-25.84
SHIFT_DY=-13.76

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