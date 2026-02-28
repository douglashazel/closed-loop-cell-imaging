#!/bin/bash

# Paths to scripts
SCRIPT0="config.py"
SCRIPT1="segmentation_monitor.py"
SCRIPT2="decision_monitor.py"
SCRIPT3="final_decision_monitor.py"
SCRIPT4="performance_monitor.py"

# Ensure scripts exist
if [[ ! -f "$SCRIPT0" || ! -f "$SCRIPT1" || ! -f "$SCRIPT2" || ! -f "$SCRIPT3" || ! -f "$SCRIPT4" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

# Combined log file
LOGFILE="/mnt/data/Close_Loop_Data/monitoring.log"

python3 -u "$SCRIPT0" 2>&1 | tee -a "$LOGFILE" &
PID0=$!

sleep 5 # wait for folders to be created

echo "Starting monitoring scripts..." | tee -a "$LOGFILE"

# Run segmentation monitoring
python3 -u "$SCRIPT1" 2>&1 | tee -a "$LOGFILE" &
PID1=$!

# Run decision monitoring
python3 -u "$SCRIPT2" 2>&1 | tee -a "$LOGFILE" &
PID2=$!

# Run decision monitoring
python3 -u "$SCRIPT3" 2>&1 | tee -a "$LOGFILE" &
PID3=$!

# Run performance monitoring
python3 -u "$SCRIPT4" 2>&1 | tee -a "$LOGFILE" &
PID4=$!

# Wait for all to finish
wait $PID0 $PID1 $PID2 $PID3 $PID4