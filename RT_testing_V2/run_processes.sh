#!/bin/bash
set -euo pipefail

# Paths to scripts
SCRIPT1="segmentation.py"
SCRIPT2="trajectories.py"

# Ensure scripts exist
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

# Run segmentation monitoring
echo "Starting cellpose segmentation..."
python3 "$SCRIPT1" &
PID1=$!

sleep 5 # wait for folders to be created

# Run trajectory processing
echo "Starting trajectory processing..."
python3 "$SCRIPT2" &
PID2=$!

# Wait for both to finish (they run indefinitely unless killed)
wait $PID1 $PID2