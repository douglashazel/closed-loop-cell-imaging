#!/bin/bash

# Paths to scripts
SCRIPT1="segmentation_monitor.py"
SCRIPT2="decision_monitor.py"
SCRIPT3="performance_monitor.py"

# Ensure scripts exist
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" || ! -f "$SCRIPT3" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

# Run segmentation monitoring
echo "Starting segmentation monitoring..."
python3 "$SCRIPT1" &
PID1=$!

sleep 5 # wait for folders to be created

# Run performance monitoring
echo "Starting performance monitoring..."
python3 "$SCRIPT3" &
PID2=$!

sleep 5 # wait for folders to be created

# Run decision monitoring
echo "Starting decision monitoring..."
python3 "$SCRIPT2" &
PID3=$!

# Wait for all to finish (they run indefinitely unless killed)
wait $PID1 $PID2 $PID3