#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="ht29_naoh_2/frames"
MASK_DIR="ht29_naoh_2/masks"
SAVE_PATH="ht29_naoh_2/analysis"

SCRIPT2="trajectories.py"

SHIFT_FRAME=2
SHIFT_DX=1.2582
SHIFT_DY=-5.1693

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