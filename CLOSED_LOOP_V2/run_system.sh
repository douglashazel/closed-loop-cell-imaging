#!/bin/bash

# Run from wherever — always works from the V6/ directory
cd "$(dirname "$0")" || exit 1

# -----------------------------
# PATHS
# -----------------------------
LOGFILE="/mnt/data/Close_Loop_Data/monitoring.log"
SCRIPTS=(config.py HandleSegmentations.py CreateDecisions.py SendDecisions.py MonitorPerformance.py)

# Ensure all scripts exist before doing anything
for s in "${SCRIPTS[@]}"; do
    [[ -f "$s" ]] || { echo "Missing script: $s"; exit 1; }
done

# Ensure log directory exists
mkdir -p "$(dirname "$LOGFILE")"

# -----------------------------
# Initialize: run config.py synchronously so directories + config.json
# are ready before any monitor starts
# -----------------------------
echo ">>> Initializing config... <<<" | tee -a "$LOGFILE"
python3 -u config.py 2>&1 | tee -a "$LOGFILE"

# -----------------------------
# Cleanup: kill all child monitors on exit / Ctrl-C
# -----------------------------
cleanup() {
    trap - EXIT INT TERM  # prevent re-entry
    echo ">>> Shutting down pipeline... <<<" | tee -a "$LOGFILE"
    # pkill -P kills every direct child of this shell (both python3 and tee
    # in each pipeline are direct children, so both die).
    pkill -TERM -P $$ 2>/dev/null
    sleep 1
    pkill -KILL -P $$ 2>/dev/null  # force-kill anything still alive
}
trap cleanup EXIT INT TERM

# -----------------------------
# Launch pipeline monitors in parallel
# -----------------------------
echo ">>> Launching pipeline... <<<" | tee -a "$LOGFILE"

# Read continuous_segmentation flag from config.json
CONTINUOUS_SEG=$(python3 -c "import json; print(json.load(open('config.json')).get('continuous_segmentation', True))")

# Segmentation — skipped entirely when continuous_segmentation=False
# (preprocess.ipynb owns frame-0 masks in that mode, so there is nothing to do)
PID1=""
if [[ "${CONTINUOUS_SEG,,}" == "true" ]]; then
    python3 -u HandleSegmentations.py 2>&1 | tee -a "$LOGFILE" &
    PID1=$!
else
    echo "continuous_segmentation=False — not launching HandleSegmentations/Cellpose" | tee -a "$LOGFILE"
fi

# Decision creation
python3 -u CreateDecisions.py 2>&1 | tee -a "$LOGFILE" &
PID2=$!

# Decision sending (ONIX)
python3 -u SendDecisions.py 2>&1 | tee -a "$LOGFILE" &
PID3=$!

# Performance monitoring
python3 -u MonitorPerformance.py 2>&1 | tee -a "$LOGFILE" &
PID4=$!

# Wait for all monitors (runs until killed)
wait $PID1 $PID2 $PID3 $PID4 2>/dev/null