#!/bin/bash
set -euo pipefail

# =============================================================================
# PLOTTING half of the preprint-figures pipeline: render every figure from the
# analysis cache (results/analysis_cache/) into
# results/<experiment>/ as standalone single-axis PNGs (the four
# responder_diagnostic figures are the multi-panel exception). Run AFTER
# run_analysis.sh has produced the caches.
#
# Rendering is fast and matplotlib's Agg backend is single-process friendly, so
# this runs sequentially (no per-experiment workers, no thread capping).
#
# Run from the project root:
#     ./run_plots.sh
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
# c2c12_chambers_dff_stack = Chamber A/B/C dF/F0 traces over the pooled
# responder mean (c2c12 only; skipped for the other experiments).
# c2c12_learning_scores = 4×2 learning panel: habituation / sensitization /
# anticipation train1 / train2, each row a score histogram + its permutation
# test (c2c12 only; restricted via the mosaic's "experiments" key).
# dmso_responder_overview = 2×2 cross-experiment panel: pooled responder-mean
# trace + stim-#8 mean response, c2c12 (top) over pc3 (bottom).
# nrk_chambers_hw_log = 2×2 NRK hardware-feedback luminosity logs: channel 1
# (chambers A, C) over channel 2 (chambers B, D) (NRK only).
# nrk_chambers_dff_corr = 4×2 NRK per-chamber panel: each chamber's dF/F0 trace
# (left) beside its Pearson-r vs distance scatter (right), one chamber per row
# (A, C, B, D); no shared y-axis (NRK only).
# c2c12_ch3_dff_pair = 1×2 bare channel-3 panel: corrected fluorescence (left) +
# normalized dF/F0 (right); no title/legend/ticks/letters, x clipped to 90 min
# (c2c12 only).
MOSAICS="c2c12_chambers_dff_stack c2c12_corr_pca_responses c2c12_learning_scores dmso_responder_overview c2c12_ch3_dff_pair nrk_chambers_hw_log nrk_chambers_dff_corr"

AGGREGATE_PDF=false       # rebuild results/preprint_figures.pdf at the end
# ─────────────────────────────────────────────────────────────────────────────


# ─────── ORCHESTRATION ───────────────────────────────────────────────────────
# This script lives at the project root; run paths are relative to it.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

CMD=(python3 SCRIPTS/preprint_analysis/make_figures.py --experiments)
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

echo "=== preprint_analysis PLOTTING pipeline ==="
echo "  ${CMD[*]}"
echo
"${CMD[@]}"

if [ "$AGGREGATE_PDF" = "true" ]; then
    if [ -f "SCRIPTS/preprint_analysis/aggregate_preprint_pdf.py" ]; then
        echo
        echo ">>> Aggregating PDF"
        python3 SCRIPTS/preprint_analysis/aggregate_preprint_pdf.py
    else
        echo "WARNING: SCRIPTS/preprint_analysis/aggregate_preprint_pdf.py not found — skipping PDF aggregation."
    fi
fi

echo
echo "=== Done. Figures in results/<experiment>/ ==="
