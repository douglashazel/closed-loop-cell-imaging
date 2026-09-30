#!/bin/bash
# Stage 1, segmentation only: Cellpose masks for every frame of one experiment.
#
#   bash run_segmentation.sh <experiment_dir | config.yaml>
#
# Parameters come from the segmentation section of
# <experiment_dir>/pipeline_config.yaml (see run_processes.sh)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_segmentation.sh <experiment_dir | config.yaml>" >&2
    exit 2
fi

# -----------------------------
# Parameters
# -----------------------------
PARAMS=$(python3 "$ROOT/SCRIPTS/core_pipeline/pipeline_config.py" prepare "$1" run_segmentation.sh segmentation)
eval "$PARAMS"
cd "$ROOT"

echo "--- Accessing ${EXP_DIR} ---"
echo "Parameters from ${CONFIG_FILE}; saved to ${RUN_RECORD}"

# -----------------------------
# Run segmentation
# -----------------------------
echo ">>> STAGE: SEGMENTATION <<<"
python3 -u SCRIPTS/core_pipeline/segmentation.py \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --flow_threshold "$FLOW_THRESHOLD" \
    --cellprob_threshold "$CELLPROB_THRESHOLD" \
    --niter "$NITER" \
    --diameter "$DIAMETER"

echo ">>> DONE <<<"
