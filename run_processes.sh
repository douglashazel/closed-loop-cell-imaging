#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/frames" # where the images are located
MASK_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/masks" # where you want to save the masks
SAVE_PATH="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/analysis" # where you want to save the analysis results
FLOW_THRESHOLD=0.9975 #default 0.4
CELLPROB_THRESHOLD=-5 #default 0.0
NITER=40000 #default 200
DIAMETER=24

SHIFT_FRAME=46      # frame where shift occurs
SHIFT_DX=1 #x displacement
SHIFT_DY=-2 #y displacement

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