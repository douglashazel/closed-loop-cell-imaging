#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/frames"
MASK_DIR="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/masks"
SAVE_PATH="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/analysis"

SCRIPT2="trajectories_optimized.py"

SHIFT_FRAME=45
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
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy "$SHIFT_DX" "$SHIFT_DY"