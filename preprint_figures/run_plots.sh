#!/bin/bash
set -euo pipefail

# =============================================================================
# PLOTTING half of the preprint-figures pipeline: render every figure from the
# analysis cache (May29_preprint_figures/analysis_cache/) into
# May29_preprint_figures/<experiment>/ as standalone single-axis PNGs (the four
# responder_diagnostic figures are the multi-panel exception). Run AFTER
# run_analysis.sh has produced the caches.
#
# Rendering is fast and matplotlib's Agg backend is single-process friendly, so
# this runs sequentially (no per-experiment workers, no thread capping).
#
# Run from the project root:
#     ./preprint_figures/run_plots.sh
# =============================================================================

# ─────── CONFIG ──────────────────────────────────────────────────────────────
# "all" or a space-separated subset of analysis groups (the make_figures
# --figures values): dff clustering average_peak correlation_distance
# response_violins learning_scores nrk_hardware_log responder_diagnostic
FIGURES="all"

# "all" or a space-separated subset of experiment names.
EXPERIMENTS="all"
# Available: c2c12_dmso_09APR26 pc3_dmso_23MAR26 nrk_acid_13APR26

# Space-separated mosaic names to assemble (see plots/mosaics.py MOSAICS), or
# empty for none. e.g. MOSAICS="dff_stack_ch1 clustering_overview"
MOSAICS=""

AGGREGATE_PDF=false       # rebuild May29_preprint_figures.pdf at the end
# ─────────────────────────────────────────────────────────────────────────────


# ─────── ORCHESTRATION ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CMD=(python3 preprint_figures/make_figures.py --experiments)
if [ "$EXPERIMENTS" = "all" ]; then
    CMD+=(all)
else
    # shellcheck disable=SC2206
    CMD+=($EXPERIMENTS)
fi
CMD+=(--figures)
if [ "$FIGURES" = "all" ]; then
    CMD+=(all)
else
    # shellcheck disable=SC2206
    CMD+=($FIGURES)
fi
if [ -n "$MOSAICS" ]; then
    # shellcheck disable=SC2206
    CMD+=(--mosaics $MOSAICS)
fi

echo "=== preprint_figures PLOTTING pipeline ==="
echo "  ${CMD[*]}"
echo
"${CMD[@]}"

if [ "$AGGREGATE_PDF" = "true" ]; then
    if [ -f "aggregate_preprint_pdf.py" ]; then
        echo
        echo ">>> Aggregating PDF"
        python3 aggregate_preprint_pdf.py
    else
        echo "WARNING: aggregate_preprint_pdf.py not found — skipping PDF aggregation."
    fi
fi

echo
echo "=== Done. Figures in May29_preprint_figures/<experiment>/ ==="
