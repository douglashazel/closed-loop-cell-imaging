#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/frames"
MASK_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/masks"
SAVE_PATH="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/analysis"

SCRIPT2="trajectories.py"

SHIFT_FRAME=46
SHIFT_DX=1
SHIFT_DY=-2

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