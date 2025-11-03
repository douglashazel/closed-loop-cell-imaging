#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="hela_dmso_2/frames"
MASK_DIR="hela_dmso_2/masks"
SAVE_PATH="hela_dmso_2/analysis"
FLOW_THRESHOLD=1.3 #default 0.4
CELLPROB_THRESHOLD=-1.55 #default 0.0
NITER=400 #default 200
DIAMETER=27

SHIFT_FRAME=2      # frame where shift occurs
SHIFT_DX=-25.5992 # x displacement
SHIFT_DY=-12.2778 # y displacement

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