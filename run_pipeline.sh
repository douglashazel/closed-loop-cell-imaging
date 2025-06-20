#!/bin/bash
# Script to automate running a sequence of Python scripts for data processing
# Exit on any error
set -e

# Define script paths
SETUP="APRIL_setup.py"
MASK_COUNT="APRIL_mask_count.py"
INFILTRATE="APRIL_infiltrate.py"
EXTRACT_BGROUNDS="APRIL_extract_bgrounds.py"
EXTRACT_CELLS="APRIL_extract_cells.py"
EXTRACT_SPIKES="APRIL_extract_spikes.py"
EXTRACT_FLASHES="APRIL_extract_flashes.py"

# Function to run a script and handle errors
run_script() {
    local script_name=$1
    shift
    echo "Started $script_name"
    if ! python3 "$script_name" "$@"; then
        echo "Error in $script_name, continuing..."
    fi
    echo
}

# 0: Setup experiment paths
run_script "$SETUP"

# 1: Count total cell masks
run_script "$MASK_COUNT"

# 2: Track cell masks across frames
# Extract movie list from parameters/setup.npy
if [ -f "parameters/setup.npy" ]; then
    # Use Python to extract movie list from setup.npy
    movie_list=$(python3 -c "import numpy as np; params = np.load('parameters/setup.npy', allow_pickle=True).item(); print(' '.join(params.keys()))")
    subprocess_num=2
    for movie in $movie_list; do
        echo "Started $INFILTRATE for $movie"
        if ! python3 "$INFILTRATE" 0 "$subprocess_num" "$movie"; then
            echo "Error in $INFILTRATE for $movie, continuing..."
        fi
        echo
    done
else
    echo "Skip $INFILTRATE: setup.npy not found"
    echo
fi

# 3: Extract background luminosity
run_script "$EXTRACT_BGROUNDS"

# 4: Extract cell luminosity
run_script "$EXTRACT_CELLS"

# 5: Find spikes
run_script "$EXTRACT_SPIKES"

# 6: Find flashes
run_script "$EXTRACT_FLASHES"

echo "Pipeline completed"