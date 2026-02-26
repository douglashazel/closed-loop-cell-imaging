#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/other/DMSO_TEST"
IMAGE_DIR="${GLOBAL_DIR}/frames"
MASK_DIR="${GLOBAL_DIR}/masks"

SCRIPT1="SCRIPTS/segmentation.py"

# -----------------------------
# CELLPOSE PARAMETERS (determine using preprocess.ipynb or the cellpose GUI)
# -----------------------------
FLOW_THRESHOLD=0.955
CELLPROB_THRESHOLD=-3
NITER=10000
DIAMETER=10

# -----------------------------
# Ensure script exists
# -----------------------------
if [[ ! -f "$SCRIPT1" ]]; then
    echo "Script $SCRIPT1 not found."
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
    --diameter "$DIAMETER"