#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
EXP="EXPERIMENTS/other/NRK_TEST"
STIM_FRAME="46"
IMAGE_DIR="${EXP}/frames"
MASK_DIR="${EXP}/masks"
ANALYSIS_DIR="${EXP}/analysis"

# -----------------------------
# Run background normalization
# -----------------------------
echo ">>> Running background normalization for ${EXP}"
python SCRIPTS/NormBGround.py \
    --exp "$EXP" \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --analysis_dir "$ANALYSIS_DIR"

# -----------------------------
# Run stimulus delta
# -----------------------------
echo ">>> Running stimulus delta computation for ${EXP} (stim_frame=${STIM_FRAME})"
python SCRIPTS/StimDelta.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR" \
    --stim_frame "$STIM_FRAME"

# -----------------------------
# Correlation analysis
# -----------------------------
echo ">>> Running pairwise correlation analysis for ${EXP}"
python SCRIPTS/Correlations.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR"

# -----------------------------
# Correlation window analysis
# -----------------------------
echo ">>> Running sliding window correlation analysis for ${EXP}"
python SCRIPTS/CorrelationWindows.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR" \
    --window_size 500 \
    --step_size 500

# -----------------------------
# Derivative and std measurements
# -----------------------------
echo ">>> Running derivative and std measurements for ${EXP}"
python SCRIPTS/DerivativeSTD.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR"

echo ">>> Pipeline finished for ${EXP}"
