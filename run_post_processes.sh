#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
EXP="u87_carbachol_1"
STIM_FRAME="5"
IMAGE_DIR="${EXP}/frames"
MASK_DIR="${EXP}/masks"
ANALYSIS_DIR="${EXP}/analysis"

# -----------------------------
# Run background normalization
# -----------------------------
echo ">>> Running background normalization for ${EXP}"
python bground_normalization.py \
    --exp "$EXP" \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --analysis_dir "$ANALYSIS_DIR"

# -----------------------------
# Run stimulus delta
# -----------------------------
echo ">>> Running stimulus delta computation for ${EXP} (stim_frame=${STIM_FRAME})"
python stimulus_delta.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR" \
    --stim_frame "$STIM_FRAME"

echo ">>> Pipeline finished for ${EXP}"