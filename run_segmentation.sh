#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
IMAGE_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/frames"
MASK_DIR="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited/masks"
FLOW_THRESHOLD=0.9975
CELLPROB_THRESHOLD=-5
NITER=40000
DIAMETER=24

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