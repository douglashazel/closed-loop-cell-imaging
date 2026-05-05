#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="EXPERIMENTS/ht29/ht29_serotonin_1"
IMAGE_DIR="${GLOBAL_DIR}/frames"
ANALYSIS_DIR="${GLOBAL_DIR}/analysis"

# -----------------------------
# ANALYSIS PARAMETERS
# -----------------------------
F0_FRAME=1
#STIM_FRAMES="77,112,148,183,218,322,357,390,424,458" # C2C12 DMSO
STIM_FRAMES="2" # "0,9,18,26,35,45,56,65,74,83,141,152,164,176,189,200,215" # NRK closed-loop March27 w/ initial setpoint

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
