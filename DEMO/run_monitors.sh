#!/bin/bash

# Paths to scripts
SCRIPT1="segmentation_monitor.py"
SCRIPT2="decision_monitor.py"

# Ensure scripts exist
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" ]]; then
    echo "One or both scripts not found."
    exit 1
fi

# Run both scripts in background
echo "Starting segmentation monitoring..."
python3 "$SCRIPT1" &
PID1=$!

sleep 5 # wait for folders to be created

echo "Starting decision monitoring..."
python3 "$SCRIPT2" &
PID2=$!

# Wait for both to finish (they run indefinitely unless killed)
wait $PID1 $PID2