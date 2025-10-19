#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="c2c12_thapsigargin_2/frames"
MASK_DIR="c2c12_thapsigargin_2/masks"
SAVE_PATH="c2c12_thapsigargin_2/analysis"
FLOW_THRESHOLD=0.89 #default 0.4
CELLPROB_THRESHOLD=-6 #default 0.0
NITER=270 #default 200
DIAMETER=13

SHIFT_FRAME=5      # frame where shift occurs
SHIFT_DX=1.7487     # x displacement
SHIFT_DY=-0.5875      # y displacement

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
    --save_path "$SAVE_PATH" \
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy "$SHIFT_DX" "$SHIFT_DY" &
PID2=$!

wait $PID1 $PID2