#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel 1/frames"
MASK_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel 1/masks"
FLOW_THRESHOLD=0.9715
CELLPROB_THRESHOLD=-6
NITER=7000
DIAMETER=29

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