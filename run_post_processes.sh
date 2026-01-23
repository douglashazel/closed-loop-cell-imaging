#!/bin/bash
set -euo pipefail

# -----------------------------
# User parameters
# -----------------------------
EXP="DMSO_C2C12_repeat_pulse_16JAN26_take2/channel 1"
STIM_FRAME="42"
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
python stimulus_delta_snr.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR" \
    --stim_frame "$STIM_FRAME"

# -----------------------------
# Correlation plotting
# -----------------------------
echo ">>> Running pairwise distance correlation for ${EXP}"
python correlation_plots.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR" \

# -----------------------------
# Correlation window plotting
# -----------------------------
echo ">>> Running stimulus delta computation for ${EXP}"
python correlation_window_plots.py \
    --exp "$EXP" \
    --analysis_dir "$ANALYSIS_DIR" \

echo ">>> Pipeline finished for ${EXP}"