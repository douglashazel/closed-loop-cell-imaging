#!/bin/bash
set -euo pipefail

cd "/mnt/exDisk1/douglashazel/DHcode/Patrick_Cell_Analysis"

GLOBAL_DIR="EXPERIMENTS/nrk/nrk_tropisetron_1"
IMAGE_DIR="${GLOBAL_DIR}/frames"
MASK_DIR="${GLOBAL_DIR}/masks"
SAVE_PATH="${GLOBAL_DIR}/analysis"

SCRIPT1="SCRIPTS/segmentation.py"
SCRIPT2="SCRIPTS/trajectories.py"

FLOW_THRESHOLD=0.965
CELLPROB_THRESHOLD=-4.0
NITER=5000
DIAMETER=27

MAX_DISTANCE=35.0
GRACE_PERIOD=3
RADIUS=999999999
RADIUS_Y=-533
RADIUS_X=309
SHIFT_FRAME=2
SHIFT_XY="0 -11"
SAVE_INTERVAL=500
RUN_MODE="full"

mkdir -p "$SAVE_PATH"
CONFIG_FILE="${SAVE_PATH}/config.txt"
cat > "$CONFIG_FILE" <<CFGEOF
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
CFGEOF
echo "Config saved to $CONFIG_FILE"

echo "--- Accessing ${GLOBAL_DIR} ---"
echo "Run mode: $RUN_MODE"

if [[ "$RUN_MODE" == "preview_only" ]]; then
    echo "Preview only: config was generated, but no analysis jobs were launched."
else
    PID1=""
    if [[ "$RUN_MODE" == "full" ]]; then
        echo ">>> STAGE: SEGMENTATION <<<"
        python3 -u "$SCRIPT1" \
            --image_dir "$IMAGE_DIR" \
            --mask_dir "$MASK_DIR" \
            --flow_threshold "$FLOW_THRESHOLD" \
            --cellprob_threshold "$CELLPROB_THRESHOLD" \
            --niter "$NITER" \
            --diameter "$DIAMETER" &
        PID1=$!
        sleep 5
    else
        echo ">>> STAGE: SEGMENTATION <<<"
        echo "Skipping segmentation; using existing mask files in $MASK_DIR"
    fi

    echo ">>> STAGE: TRAJECTORIES <<<"
    python3 -u "$SCRIPT2" \
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

    if [[ -n "$PID1" ]]; then
        wait "$PID1"
    fi
    wait "$PID2"

    echo ">>> STAGE: PRE-ANALYSIS <<<"
    python3 -u SCRIPTS/PreAnalysis.py \
        --exp "$GLOBAL_DIR" \
        --analysis_dir "$SAVE_PATH"
fi

echo ">>> DONE <<<"
