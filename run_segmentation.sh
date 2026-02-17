#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/frames"
MASK_DIR="resize30perc_NRK_ArcLight_acids_05FEB26_3646_of_4374/masks"
FLOW_THRESHOLD=0.955
CELLPROB_THRESHOLD=-3
NITER=10000
DIAMETER=10

SCRIPT1="segmentation.py"

# -----------------------------
# Ensure script exists
# -----------------------------
if [[ ! -f "$SCRIPT1" ]]; then
    echo "Script $SCRIPT1 not found."
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
    --diameter "$DIAMETER"