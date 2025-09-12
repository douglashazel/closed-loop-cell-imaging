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

# Combined log file
LOGFILE="/mnt/data/Close_Loop_Data/monitoring.log"

echo "Starting monitoring scripts..." | tee -a "$LOGFILE"

# Run segmentation monitoring
python3 -u "$SCRIPT1" 2>&1 | tee -a "$LOGFILE" &
PID1=$!

sleep 5 # wait for folders to be created

# Run decision monitoring
python3 -u "$SCRIPT2" 2>&1 | tee -a "$LOGFILE" &
PID2=$!

# Run performance monitoring
python3 -u "$SCRIPT3" 2>&1 | tee -a "$LOGFILE" &
PID3=$!

# Wait for all to finish
wait $PID1 $PID2 $PID3