#!/bin/bash
set -euo pipefail

# =============================================================================
# Master orchestrator for the modular preprint-figures pipeline.
#
# Edit the CONFIG block below to choose which analyses run and on which
# experiments. Each analysis is a standalone python script under
# preprint_figures/. Background-correction state is cached per experiment to
# April28_preprint_results/bg_cache/ so repeated runs across analyses are fast.
#
# Run from the project root (so SCRIPTS/ and EXPERIMENTS/ resolve correctly):
#     ./preprint_figures/run_figures.sh
# =============================================================================

# ─────── CONFIG ──────────────────────────────────────────────────────────────
# Set to "all" or a space-separated subset of analyses.
# ANALYSES="all"
ANALYSES="clustering dff average_peak correlation_distance response_violins responder_diagnostic learning_scores nrk_hardware_log"
# Available:
#   dff                  — dF/F0 stacked traces + responder-pooled mean
#   average_peak         — per-stimulus dF/F0 peak overlay + mean (DMSO only)
#   correlation_distance — pairwise correlation vs spatial distance
#   clustering           — pooled PCA + UMAP scatter (no clustering)
#   response_violins     — pooled per-stim asymmetric violin (height + width)
#   responder_diagnostic — responder distribution + stimulus-locked artifact check
#   learning_scores      — habituation / sensitization summed scores (DMSO only)
#   nrk_hardware_log     — hardware feedback log (NRK only)

# Set to "all" or a space-separated subset of experiment names.
EXPERIMENTS="all"
# Available: c2c12_dmso_09APR26 pc3_dmso_23MAR26 nrk_acid_13APR26

RECOMPUTE_BG=false        # force background-cache rebuild for selected experiments
AGGREGATE_PDF=true        # rebuild April28_preprint_figures.pdf at the end
PARALLEL=true             # run the 3 experiments concurrently (one worker each)
# ─────────────────────────────────────────────────────────────────────────────


# ─────── ORCHESTRATION ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ALL_ANALYSES=(
    clustering
    dff
    average_peak
    correlation_distance
    response_violins
    responder_diagnostic
    learning_scores
    nrk_hardware_log
)

if [ "$ANALYSES" = "all" ]; then
    ANALYSES_LIST=("${ALL_ANALYSES[@]}")
else
    read -r -a ANALYSES_LIST <<< "$ANALYSES"
fi

# Resolve EXPERIMENTS to an explicit name list (expand "all" via the config).
if [ "$EXPERIMENTS" = "all" ]; then
    read -r -a EXP_LIST <<< "$(
        python3 -c "import sys; sys.path.insert(0, 'preprint_figures'); \
from common.config import EXPERIMENTS; print(' '.join(EXPERIMENTS))"
    )"
else
    read -r -a EXP_LIST <<< "$EXPERIMENTS"
fi

# Fail fast if any selected analysis script is missing.
for a in "${ANALYSES_LIST[@]}"; do
    if [ ! -f "preprint_figures/${a}.py" ]; then
        echo "ERROR: missing preprint_figures/${a}.py"
        exit 1
    fi
done

echo "=== preprint_figures pipeline ==="
echo "  analyses:    ${ANALYSES_LIST[*]}"
echo "  experiments: ${EXP_LIST[*]}"
echo "  parallel=${PARALLEL}, recompute_bg=${RECOMPUTE_BG}, aggregate_pdf=${AGGREGATE_PDF}"
echo

# Run every selected analysis for ONE experiment, in order. The first analysis
# carries --recompute-bg (if requested) so the background cache is rebuilt
# once; later analyses for the same experiment hit the warm cache.
run_experiment() {
    local exp="$1"
    local first=true
    for a in "${ANALYSES_LIST[@]}"; do
        local extra=()
        if [ "$first" = "true" ] && [ "$RECOMPUTE_BG" = "true" ]; then
            extra+=(--recompute-bg)
        fi
        first=false
        echo ">>> [${exp}] Running ${a}"
        python3 "preprint_figures/${a}.py" --experiments "$exp" "${extra[@]}"
    done
    echo ">>> [${exp}] done"
}

if [ "$PARALLEL" = "true" ] && [ "${#EXP_LIST[@]}" -gt 1 ]; then
    # One concurrent worker per experiment. Cap each worker's BLAS/OpenMP
    # thread pool so the 3 workers don't oversubscribe the CPU.
    NWORKERS="${#EXP_LIST[@]}"
    THREADS_PER_WORKER=$(( $(nproc) / NWORKERS ))
    [ "$THREADS_PER_WORKER" -lt 1 ] && THREADS_PER_WORKER=1
    export OMP_NUM_THREADS="$THREADS_PER_WORKER" \
           OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER" \
           MKL_NUM_THREADS="$THREADS_PER_WORKER" \
           NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"

    LOG_DIR="April28_preprint_results/run_logs"
    mkdir -p "$LOG_DIR"
    echo ">>> Launching ${NWORKERS} workers (${THREADS_PER_WORKER} threads each); logs in ${LOG_DIR}/"

    PIDS=()
    for exp in "${EXP_LIST[@]}"; do
        run_experiment "$exp" > "${LOG_DIR}/${exp}.log" 2>&1 &
        PIDS+=("$!")
    done

    # Wait for all workers; replay each log; fail if any worker errored.
    FAIL=0
    for i in "${!PIDS[@]}"; do
        rc=0
        wait "${PIDS[i]}" || rc=$?
        exp="${EXP_LIST[i]}"
        echo
        echo "───── ${exp} (exit ${rc}) ─────"
        cat "${LOG_DIR}/${exp}.log"
        [ "$rc" -ne 0 ] && FAIL=1
    done
    if [ "$FAIL" -ne 0 ]; then
        echo
        echo "ERROR: one or more experiment workers failed (see logs above)."
        exit 1
    fi
else
    # Sequential fallback: one experiment at a time.
    for exp in "${EXP_LIST[@]}"; do
        run_experiment "$exp"
        echo
    done
fi

if [ "$AGGREGATE_PDF" = "true" ]; then
    if [ -f "aggregate_preprint_pdf.py" ]; then
        echo ">>> Aggregating PDF"
        python3 aggregate_preprint_pdf.py
    else
        echo "WARNING: aggregate_preprint_pdf.py not found — skipping PDF aggregation."
    fi
fi

echo
echo "=== Done. Figures in April28_preprint_results/<experiment>/ ==="
