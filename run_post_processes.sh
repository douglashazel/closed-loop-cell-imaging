#!/bin/bash
# Stage 1 quick-look post-analysis for one experiment: background correction,
# derivative/STD, dF/F0 plots.
#
#   bash run_post_processes.sh <experiment_dir | config.yaml>
#
# Parameters come from the post_analysis section of
# <experiment_dir>/pipeline_config.yaml (see run_processes.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_post_processes.sh <experiment_dir | config.yaml>" >&2
    exit 2
fi

# -----------------------------
# Parameters (stops here if the config is missing or invalid)
# -----------------------------
PARAMS=$(python3 "$ROOT/SCRIPTS/core_pipeline/pipeline_config.py" prepare "$1" run_post_processes.sh post_analysis)
eval "$PARAMS"
cd "$ROOT"

echo "--- Accessing ${EXP_DIR} ---"
echo "Parameters from ${CONFIG_FILE}; saved to ${RUN_RECORD}"

# -----------------------------
# Post-analysis: background correction, derivative/STD, dF/F0
# -----------------------------
echo ">>> STAGE: POST-ANALYSIS <<<"
python3 -u SCRIPTS/core_pipeline/PostAnalysis.py \
    --exp "$EXP_DIR" \
    --image_dir "$IMAGE_DIR" \
    --analysis_dir "$ANALYSIS_DIR" \
    --f0_frame "$F0_FRAME" \
    --stim_frames "$STIM_FRAMES"

echo ">>> DONE <<<"
