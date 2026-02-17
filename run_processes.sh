#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/frames" # where the images are located
MASK_DIR="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/masks" # where you want to save the masks
SAVE_PATH="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/analysis" # where you want to save the analysis results
FLOW_THRESHOLD=0.955 #default 0.4
CELLPROB_THRESHOLD=-3 #default 0.0
NITER=10000 #default 200
DIAMETER=10

SHIFT_FRAME=46      # frame where shift occurs
SHIFT_DX=0 #x displacement
SHIFT_DY=0 #y displacement

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