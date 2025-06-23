#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default values for the first script
INPUT_FOLDER="/mnt/data/pc3_naoh_channel1_30JAN25"
CELLPROB="0.0"
FLOWTHRESHOLD="0.4"
DIAMETER="47"
CORES="10"
SAVE_PATH="/mnt/data/pc3_naoh_channel1_30JAN25"
PARTITIONS="15"
FRAME_CHOICE="002"

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options for Cellpose:"
    echo "  -i INPUT_FOLDER"
    echo "  -c CELLPROB"
    echo "  -f FLOWTHRESHOLD"
    echo "  -d DIAMETER"
    echo "  -n CORES"
    echo ""
    echo "Options for downstream processing:"
    echo "  -p SAVE_PATH"
    echo "  -t PARTITIONS"
    exit 1
}

# Parse command-line arguments
while getopts "i:c:f:d:n:p:t:f:" opt; do
    case $opt in
        i) INPUT_FOLDER="$OPTARG" ;;
        c) CELLPROB="$OPTARG" ;;
        f) FLOWTHRESHOLD="$OPTARG" ;;
        d) DIAMETER="$OPTARG" ;;
        n) CORES="$OPTARG" ;;
        p) SAVE_PATH="$OPTARG" ;;
        t) PARTITIONS="$OPTARG" ;;
        f) FRAME_CHOICE="$OPTARG" ;;  # Add frame_choice option
        *) usage ;;
    esac
done

# # Step 0: Run Cellpose processing
# echo "Running Cellpose..."
# CELPOSE_CMD="python run_cellpose.py \"$INPUT_FOLDER\" --cellprob $CELLPROB --flowthreshold $FLOWTHRESHOLD --diameter $DIAMETER --cores $CORES"
# echo "Command: $CELPOSE_CMD"
# eval $CELPOSE_CMD

# echo "Running arr_to_dict..."
# ARR_TO_DICT_CMD="python arr_to_dict.py \"$INPUT_FOLDER\""
# echo "Command: $ARR_TO_DICT_CMD"
# eval $ARR_TO_DICT_CMD

# # Step 1: Run infiltrate_mod with parallel processing
# echo "Running infiltrate_dynamic..."
# for PART in $(seq 0 $((PARTITIONS - 1))); do
#     ###INFILTRATE_MOD_CMD="python infiltrate_dynamic.py $PART $PARTITIONS --directory_path \"$INPUT_FOLDER\" --save_path \"$SAVE_PATH\""
#     INFILTRATE_MOD_CMD="python dynamic_infiltrate_v2.py $PART $PARTITIONS --directory_path \"$INPUT_FOLDER\" --save_path \"$SAVE_PATH\"" # new line from Doug 6-9-25
#     echo "Command: $INFILTRATE_MOD_CMD"
#     eval $INFILTRATE_MOD_CMD &
# done
# wait

# Step 2: Run extract_bground
echo "Running extract_bground..."
EXTRACT_CMD="python extract_bground_JUNE20.py \"$INPUT_FOLDER\""
echo "Command: $EXTRACT_CMD"
eval $EXTRACT_CMD

# Step 3: Run extract_dynamic
echo "Running extract_dynamic..."
for PART in $(seq 0 $((PARTITIONS - 1))); do
    EXTRACT_CMD="python extract_dynamic_JUNE20.py $PART $PARTITIONS \"$INPUT_FOLDER\" \"$FRAME_CHOICE\""
    echo "Command: $EXTRACT_CMD"
    eval $EXTRACT_CMD &
done
wait

# Step 4: Combine luminosity data
echo "Running combine_luminosity..."
COMBINE_CMD="python luminosity_vals.py --root_dir \"$SAVE_PATH\""
echo "Command: $COMBINE_CMD"
eval $COMBINE_CMD

# Step 5: Plot time series data
echo "Running plot_data..."
PLOT_CMD="python plot_data_JUNE20.py --root_dir \"$SAVE_PATH\""
echo "Command: $PLOT_CMD"
eval $PLOT_CMD

echo "All steps completed successfully."