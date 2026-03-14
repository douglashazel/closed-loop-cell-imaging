#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/other/NRK_doug"
IMAGE_DIR="${GLOBAL_DIR}/frames" # where the images are located
MASK_DIR="${GLOBAL_DIR}/masks"  # where you want to save the masks
SAVE_PATH="${GLOBAL_DIR}/analysis" # where you want to save the analysis results

SCRIPT1="SCRIPTS/segmentation.py"
SCRIPT2="SCRIPTS/trajectories.py"

# -----------------------------
# CELLPOSE PARAMETERS (determine using preprocess.ipynb or the cellpose GUI)
# -----------------------------
FLOW_THRESHOLD=0.9975
CELLPROB_THRESHOLD=-5
NITER=40000
DIAMETER=24

# -----------------------------
# TRAJECTORY PARAMETERS (determine using preprocess.ipynb)
# -----------------------------
MAX_DISTANCE=20
GRACE_PERIOD=3
RADIUS=340
RADIUS_Y=53
RADIUS_X=-5
SHIFT_FRAME=45
SHIFT_XY="0 0"
SAVE_INTERVAL=1000

# -----------------------------
# Ensure scripts exist
# -----------------------------
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

# -----------------------------
# Log config
# -----------------------------
mkdir -p "$SAVE_PATH"
CONFIG_FILE="${SAVE_PATH}/config.txt"
cat > "$CONFIG_FILE" <<EOF
Run date: $(date)

[PATHS]
GLOBAL_DIR=$GLOBAL_DIR
IMAGE_DIR=$IMAGE_DIR
MASK_DIR=$MASK_DIR
SAVE_PATH=$SAVE_PATH

[CELLPOSE]
FLOW_THRESHOLD=$FLOW_THRESHOLD
CELLPROB_THRESHOLD=$CELLPROB_THRESHOLD
NITER=$NITER
DIAMETER=$DIAMETER

[TRAJECTORIES]
MAX_DISTANCE=$MAX_DISTANCE
GRACE_PERIOD=$GRACE_PERIOD
RADIUS=$RADIUS
RADIUS_Y=$RADIUS_Y
RADIUS_X=$RADIUS_X
SHIFT_FRAME=$SHIFT_FRAME
SHIFT_XY=$SHIFT_XY
SAVE_INTERVAL=$SAVE_INTERVAL
EOF
echo "Config saved to $CONFIG_FILE"

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
    --max_distance "$MAX_DISTANCE" \
    --grace_period "$GRACE_PERIOD" \
    --radius "$RADIUS" \
    --radius_y "$RADIUS_Y" \
    --radius_x "$RADIUS_X" \
    --shift_frame "$SHIFT_FRAME" \
    --shift_xy $SHIFT_XY \
    --save_interval "$SAVE_INTERVAL" &
PID2=$!

wait $PID1 $PID2

# -----------------------------
# Pre-analysis plots
# -----------------------------
echo ">>> Running pre-analysis plots for ${GLOBAL_DIR}"
python3 SCRIPTS/PreAnalysis.py \
    --exp "$GLOBAL_DIR" \
    --analysis_dir "$SAVE_PATH"