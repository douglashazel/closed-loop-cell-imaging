#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/other/DMSO_C2C12_repeat_kat_2"
IMAGE_DIR="${GLOBAL_DIR}/frames"
MASK_DIR="${GLOBAL_DIR}/masks"
ANALYSIS_DIR="${GLOBAL_DIR}/analysis"

# -----------------------------
# ANALYSIS PARAMETERS
# -----------------------------
STIM_FRAME=45 # frame shift
WINDOW_SIZE=100 # tune according to experiment length (e.g. 500 for 3000 frames, 200 for 1000 frames)
STEP_SIZE=510 # tune according to experiment length and desired resolution

echo "--- Accessing ${GLOBAL_DIR} ---"
# -----------------------------
# Run background normalization
# -----------------------------
echo ">>> Running background normalization for ${GLOBAL_DIR}"
python SCRIPTS/NormBGround.py \
    --exp "$GLOBAL_DIR" \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --analysis_dir "$ANALYSIS_DIR"

# -----------------------------
# Derivative and std measurements
# -----------------------------
echo ">>> Running derivative and std measurements for ${GLOBAL_DIR}"
python SCRIPTS/DerivativeSTD.py \
    --exp "$GLOBAL_DIR" \
    --analysis_dir "$ANALYSIS_DIR"

echo ">>> Pipeline finished for ${GLOBAL_DIR}"
