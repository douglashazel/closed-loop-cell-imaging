#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="hela_dmso_2/frames"
MASK_DIR="hela_dmso_2/masks"
FLOW_THRESHOLD=1.3
CELLPROB_THRESHOLD=-1.55
NITER=400
DIAMETER=27

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