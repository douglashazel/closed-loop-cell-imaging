#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="ht29_naoh_2/frames"
MASK_DIR="ht29_naoh_2/masks"
FLOW_THRESHOLD=0.965
CELLPROB_THRESHOLD=-6
NITER=800
DIAMETER=15

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