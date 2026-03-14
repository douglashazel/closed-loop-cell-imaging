#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/other/NRK_doug"
IMAGE_DIR="${GLOBAL_DIR}/frames"
ANALYSIS_DIR="${GLOBAL_DIR}/analysis"

# -----------------------------
# ANALYSIS PARAMETERS
# -----------------------------
F0_FRAME=36
STIM_FRAMES="77,112,148,183,218,322,357,390,424,458"

echo "--- Accessing ${GLOBAL_DIR} ---"
# -----------------------------
# Post-analysis: background correction, derivative/STD, dF/F0
# -----------------------------
echo ">>> Running post-analysis for ${GLOBAL_DIR}"
python SCRIPTS/PostAnalysis.py \
    --exp "$GLOBAL_DIR" \
    --image_dir "$IMAGE_DIR" \
    --analysis_dir "$ANALYSIS_DIR" \
    --f0_frame "$F0_FRAME" \
    --stim_frames "$STIM_FRAMES"

echo ">>> Pipeline finished for ${GLOBAL_DIR}"
