#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="c2c12_carbachol_1/frames"
MASK_DIR="c2c12_carbachol_1/masks"
SAVE_PATH="c2c12_carbachol_1/analysis"
FLOW_THRESHOLD=0.8      # default=0.4; higher = stricter flow consistency, fewer masks
CELLPROB_THRESHOLD=-1.25   # default=0.0; higher = fewer cells accepted, lower = more cells
NITER=200                # default=200; higher = slower but can improve accuracy
DIAMETER=11             # default=0

SCRIPT1="segmentation.py"
SCRIPT2="trajectories.py"

# -----------------------------
# Ensure scripts exist
# -----------------------------
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

# -----------------------------
# Run segmentation
# -----------------------------
echo "Starting cellpose segmentation..."
python3 "$SCRIPT1" \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --flow_threshold "$FLOW_THRESHOLD" \
    --cellprob_threshold "$CELLPROB_THRESHOLD" \
    --niter "$NITER" \
    --diameter "$DIAMETER" &
PID1=$!

sleep 5

# -----------------------------
# Run trajectory processing
# -----------------------------
echo "Starting trajectory processing..."
python3 "$SCRIPT2" \
    --mask_dir "$MASK_DIR" \
    --image_dir "$IMAGE_DIR" \
    --save_path "$SAVE_PATH" &
PID2=$!

wait $PID1 $PID2