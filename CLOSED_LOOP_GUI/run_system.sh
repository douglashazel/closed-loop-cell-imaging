#!/bin/bash

# Run from wherever — always works from this script's directory
cd "$(dirname "$0")" || exit 1

# -----------------------------
# PATHS
# -----------------------------
SCRIPTS=(config.py HandleSegmentations.py CreateDecisions.py SendDecisions.py MonitorPerformance.py fake_frames.py)

# Seconds SendDecisions gets to abort and close the ONIX run on shutdown
# before anything still alive is SIGKILLed (the web GUI's Stop waits longer)
STOP_GRACE_SEC=60

# Ensure all scripts exist before doing anything
for s in "${SCRIPTS[@]}"; do
    [[ -f "$s" ]] || { echo "Missing script: $s"; exit 1; }
done

# -----------------------------
# Initialize: run config.py synchronously so directories + config.json
# are ready before any monitor starts. Output goes to stdout only since
# LOGFILE is not known until after config.json exists.
# -----------------------------
echo ">>> Initializing config... <<<"
python3 -u config.py

# Read log path from config so the launcher and the GUI agree on it
LOGFILE=$(python3 -c "import json; print(json.load(open('config.json'))['log_path'])")
mkdir -p "$(dirname "$LOGFILE")"

# -----------------------------
# Cleanup: stop all child monitors on exit / Ctrl-C / Stop
# -----------------------------
cleanup() {
    trap - EXIT        # prevent re-entry
    trap '' INT TERM   # a repeated signal must not cut the ONIX abort short
    echo ">>> Shutting down pipeline... <<<" | tee -a "$LOGFILE"
    # pkill -P signals every direct child of this shell (python3 and tee in
    # each pipeline). The tees ignore it (see launch) and exit once their
    # python3 closes the pipe, so SendDecisions can still log its Abort/Close.
    pkill -TERM -P $$ 2>/dev/null
    for ((i = 0; i < STOP_GRACE_SEC; i++)); do
        pgrep -P $$ >/dev/null || break
        sleep 1
    done
    pkill -KILL -P $$ 2>/dev/null  # force-kill anything still alive
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Start one daemon in the background, appending its output to LOGFILE
launch() {
    python3 -u "$@" 2>&1 | (trap '' INT TERM; exec tee -a "$LOGFILE") &
}

# -----------------------------
# Launch pipeline monitors in parallel
# -----------------------------
echo ">>> Launching pipeline... <<<" | tee -a "$LOGFILE"

# Read continuous_segmentation and dry_run flags from config.json
CONTINUOUS_SEG=$(python3 -c "import json; print(json.load(open('config.json')).get('continuous_segmentation', True))")
DRY_RUN=$(python3 -c "import json; print(json.load(open('config.json')).get('dry_run', False))")

# Segmentation — skipped entirely when continuous_segmentation=False
# (frame-0 masks come from the web GUI's Push mask action in that mode,
# so there is nothing to do)
if [[ "${CONTINUOUS_SEG,,}" == "true" ]]; then
    launch HandleSegmentations.py
else
    echo "continuous_segmentation=False — not launching HandleSegmentations/Cellpose" | tee -a "$LOGFILE"
fi

# Decision creation
launch CreateDecisions.py

# Decision sending (ONIX, or a simulated ONIX when dry_run=True)
launch SendDecisions.py

# Performance monitoring
launch MonitorPerformance.py

# Dry run: synthetic frames stand in for the microscope
if [[ "${DRY_RUN,,}" == "true" ]]; then
    echo "dry_run=True — simulated ONIX, synthetic frames from fake_frames.py" | tee -a "$LOGFILE"
    launch fake_frames.py
fi

# Supervise: the daemons run until stopped, so any one exiting breaks the
# loop. Stop the rest (cleanup trap, which lets SendDecisions abort the
# ONIX run) rather than keep driving the hardware without it.
wait -n
echo ">>> A pipeline process exited; stopping the others <<<" | tee -a "$LOGFILE"
exit 1