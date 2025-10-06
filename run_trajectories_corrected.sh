#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="pc3_carbachol_1_corrected/frames"
MASK_DIR="pc3_carbachol_1_corrected/masks"
SAVE_PATH="pc3_carbachol_1_corrected/analysis"

SCRIPT2="trajectories.py"

SHIFT_FRAME=5
SHIFT_DX=-250
SHIFT_DY=10

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