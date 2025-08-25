#!/bin/bash

DIR="./frames_RTX40"               # Directory with .tif images
GPUS=(0)                           # GPUs to use
RUNS_PER_GPU=1                     # Runs per GPU
CELLP_CMD="python -m cellpose --dir $DIR --cellprob_threshold 0.0 --flow_threshold 0.4 --use_gpu --verbose"

for GPU_ID in "${GPUS[@]}"; do
    for ((run=0; run<RUNS_PER_GPU; run++)); do
        LOGFILE="logs_RTX40/RUN${run}_GPU${GPU_ID}.txt"
        echo "Starting RUN $run on GPU $GPU_ID"
        START=$(date +%s)

        CUDA_VISIBLE_DEVICES=$GPU_ID $CELLP_CMD >> "$LOGFILE" 2>&1

        END=$(date +%s)
        TOTAL=$((END - START))
        echo "Total time for RUN $run on GPU $GPU_ID: ${TOTAL}s" >> "$LOGFILE"

        echo "Completed RUN $run on GPU $GPU_ID (Log: $LOGFILE)"
    done
done
