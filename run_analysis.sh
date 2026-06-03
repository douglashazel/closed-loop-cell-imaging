#!/bin/bash
set -euo pipefail

# =============================================================================
# ANALYSIS half of the preprint-figures pipeline: run the analyze_*.py scripts
# to compute figure-ready intermediates and CACHE them per (experiment,
# analysis) under results/analysis_cache/<exp>/<analysis>.pkl.
# NO plotting happens here — render with run_plots.sh afterward.
#
# Background-correction state is cached per experiment in
# results/bg_cache/; repeated runs hit the warm cache and are
# fast. The shared `responders` step runs first so every consumer reads one
# deterministic responder mask (consumers also fall back to computing it).
#
# Run from the project root:
#     ./run_analysis.sh
# =============================================================================

# ─────── CONFIG ──────────────────────────────────────────────────────────────
# Set to "all" or a space-separated subset of analyses.
ANALYSES="all"
# Available (each is SCRIPTS/preprint_analysis/analyze_<name>.py):
#   responders            — shared responder thresholds + masks (run first)
#   dff                   — dF/F0 stacked traces + responder-pooled mean
#   average_peak          — per-stimulus dF/F0 peak segments (DMSO only)
#   correlation_distance  — pairwise correlation vs spatial distance + Mantel
#   clustering            — pooled PCA + UMAP embeddings (no clustering)
#   response_violins      — pooled per-stim height/width arrays + train means
#   responder_diagnostic  — responder QC panels (+ frame-image sharpness)
#   learning_scores       — habituation / sensitization / anticipation (DMSO)
#   nrk_hardware_log      — hardware feedback luminosity log (NRK only)

# Set to "all" or a space-separated subset of experiment names.
EXPERIMENTS="all"
# Available: c2c12_dmso_09APR26 pc3_dmso_23MAR26 nrk_acid_13APR26

RECOMPUTE_BG=false        # force background-cache rebuild for selected experiments
PARALLEL=true             # run the experiments concurrently (one worker each)
# ─────────────────────────────────────────────────────────────────────────────


# ─────── ORCHESTRATION ───────────────────────────────────────────────────────
# This script lives at the project root; run paths are relative to it.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# `responders` runs first so the shared responders.pkl exists before consumers.
ALL_ANALYSES=(
    responders
    dff
    average_peak
    correlation_distance
    clustering
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
        python3 -c "import sys; sys.path.insert(0, 'SCRIPTS/preprint_analysis'); \
from common.config import EXPERIMENTS; print(' '.join(EXPERIMENTS))"
    )"
else
    read -r -a EXP_LIST <<< "$EXPERIMENTS"
fi

# Fail fast if any selected analysis script is missing.
for a in "${ANALYSES_LIST[@]}"; do
    if [ ! -f "SCRIPTS/preprint_analysis/analyze_${a}.py" ]; then
        echo "ERROR: missing SCRIPTS/preprint_analysis/analyze_${a}.py"
        exit 1
    fi
done

echo "=== preprint_analysis ANALYSIS pipeline ==="
echo "  analyses:    ${ANALYSES_LIST[*]}"
echo "  experiments: ${EXP_LIST[*]}"
echo "  parallel=${PARALLEL}, recompute_bg=${RECOMPUTE_BG}"
echo

# Run every selected analysis for ONE experiment, in order. The first analysis
# carries --recompute-bg (if requested) so the background cache is rebuilt once;
# later analyses for the same experiment hit the warm cache.
run_experiment() {
    local exp="$1"
    local first=true
    for a in "${ANALYSES_LIST[@]}"; do
        local extra=()
        if [ "$first" = "true" ] && [ "$RECOMPUTE_BG" = "true" ]; then
            extra+=(--recompute-bg)
        fi
        first=false
        echo ">>> [${exp}] analyze ${a}"
        python3 "SCRIPTS/preprint_analysis/analyze_${a}.py" --experiments "$exp" "${extra[@]}"
    done
    echo ">>> [${exp}] done"
}

if [ "$PARALLEL" = "true" ] && [ "${#EXP_LIST[@]}" -gt 1 ]; then
    # One concurrent worker per experiment. Cap each worker's BLAS/OpenMP thread
    # pool so the workers don't oversubscribe the CPU.
    NWORKERS="${#EXP_LIST[@]}"
    THREADS_PER_WORKER=$(( $(nproc) / NWORKERS ))
    [ "$THREADS_PER_WORKER" -lt 1 ] && THREADS_PER_WORKER=1
    export OMP_NUM_THREADS="$THREADS_PER_WORKER" \
           OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER" \
           MKL_NUM_THREADS="$THREADS_PER_WORKER" \
           NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"

    LOG_DIR="results/run_logs"
    mkdir -p "$LOG_DIR"
    echo ">>> Launching ${NWORKERS} workers (${THREADS_PER_WORKER} threads each); logs in ${LOG_DIR}/"

    PIDS=()
    for exp in "${EXP_LIST[@]}"; do
        run_experiment "$exp" > "${LOG_DIR}/${exp}.analysis.log" 2>&1 &
        PIDS+=("$!")
    done

    FAIL=0
    for i in "${!PIDS[@]}"; do
        rc=0
        wait "${PIDS[i]}" || rc=$?
        exp="${EXP_LIST[i]}"
        echo
        echo "───── ${exp} (exit ${rc}) ─────"
        cat "${LOG_DIR}/${exp}.analysis.log"
        [ "$rc" -ne 0 ] && FAIL=1
    done
    if [ "$FAIL" -ne 0 ]; then
        echo
        echo "ERROR: one or more experiment workers failed (see logs above)."
        exit 1
    fi
else
    for exp in "${EXP_LIST[@]}"; do
        run_experiment "$exp"
        echo
    done
fi

echo
echo "=== Done. Caches in results/analysis_cache/<experiment>/ ==="
echo "    Render them with ./run_plots.sh"
