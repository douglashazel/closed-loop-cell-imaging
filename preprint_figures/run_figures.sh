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
ANALYSES="all"
# Available:
#   bg_diagnostic        — background-correction diagnostic per experiment
#   time_traces          — per-cell luminosity traces (with + without stim shading)
#   dff                  — dF/F0 + responder-pooled mean + diagnostic
#   correlation_distance — pairwise correlation vs spatial distance
#   clustering           — PCA + UMAP + KMeans (per channel + pooled + kselect)
#   response_violins     — per-stim asymmetric violin (height + width + responders)
#   learning_scores      — habituation / sensitization / anticipation (DMSO only)
#   sliding_correlation  — sliding Pearson + Spearman (slow; off by default)
#   nrk_hardware_log     — hardware feedback log (NRK only)

# Set to "all" or a space-separated subset of experiment names.
EXPERIMENTS="all"
# Available: c2c12_dmso_09APR26 pc3_dmso_23MAR26 nrk_acid_13APR26

RECOMPUTE_BG=false        # force background-cache rebuild for selected experiments
AGGREGATE_PDF=true        # rebuild April28_preprint_figures.pdf at the end
# ─────────────────────────────────────────────────────────────────────────────


# ─────── ORCHESTRATION ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# sliding_correlation is excluded from "all" by default because it is slow and
# is currently disabled in april28_final_figures.py main(). Add it explicitly
# (ANALYSES="sliding_correlation ..." or "all sliding_correlation") to run it.
ALL_ANALYSES=(
    bg_diagnostic
    time_traces
    dff
    correlation_distance
    clustering
    response_violins
    learning_scores
    nrk_hardware_log
)

if [ "$ANALYSES" = "all" ]; then
    ANALYSES_LIST=("${ALL_ANALYSES[@]}")
else
    read -r -a ANALYSES_LIST <<< "$ANALYSES"
fi

if [ "$EXPERIMENTS" = "all" ]; then
    EXP_FLAG=(--experiments all)
else
    read -r -a EXP_LIST <<< "$EXPERIMENTS"
    EXP_FLAG=(--experiments "${EXP_LIST[@]}")
fi

EXTRA_FLAGS=()
[ "$RECOMPUTE_BG" = "true" ] && EXTRA_FLAGS+=(--recompute-bg)

echo "=== preprint_figures pipeline ==="
echo "  analyses:    ${ANALYSES_LIST[*]}"
echo "  experiments: ${EXPERIMENTS}"
echo "  recompute_bg=${RECOMPUTE_BG}, aggregate_pdf=${AGGREGATE_PDF}"
echo

for a in "${ANALYSES_LIST[@]}"; do
    script="preprint_figures/${a}.py"
    if [ ! -f "$script" ]; then
        echo "ERROR: missing $script"
        exit 1
    fi
    echo ">>> Running ${a}"
    python3 "$script" "${EXP_FLAG[@]}" "${EXTRA_FLAGS[@]}"
    echo
done

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
