#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/other/DMSO_TEST"
IMAGE_DIR="${GLOBAL_DIR}/frames" # where the images are located
MASK_DIR="${GLOBAL_DIR}/masks"  # where you want to save the masks
SAVE_PATH="${GLOBAL_DIR}/analysis" # where you want to save the analysis results

SCRIPT1="SCRIPTS/segmentation.py"
SCRIPT2="SCRIPTS/trajectories.py"

# -----------------------------
# CELLPOSE PARAMETERS
# -----------------------------
FLOW_THRESHOLD=0.955 #default 0.4
CELLPROB_THRESHOLD=-3 #default 0.0
NITER=10000 #default 200
DIAMETER=10

# -----------------------------
# Ensure scripts exist
# -----------------------------
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

echo "--- Accessing ${GLOBAL_DIR} ---"
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

sleep 5 # small buffer

# -----------------------------
# Run trajectory processing
# -----------------------------
echo "Starting trajectory processing..."
python3 "$SCRIPT2" \
    --mask_dir "$MASK_DIR" \
    --image_dir "$IMAGE_DIR" \
    --save_path "$SAVE_PATH" \
    --max_distance 20 \
    --grace_period 3 \
    --radius 380 \
    --radius_y 120 \
    --radius_x -30 \
    --shift_frame 46 \
    --shift_xy 0 0 \
    --save_interval 500 &
PID2=$!

wait $PID1 $PID2