#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/frames"
MASK_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/masks"
SAVE_PATH="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/analysis"

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
    --max_distance 40.0 \
    --grace_period 3 \
    --radius 380 \
    --y_shift -30 \
    --x_shift -50 \
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy "$SHIFT_DX" "$SHIFT_DY" \
    --save_interval 500