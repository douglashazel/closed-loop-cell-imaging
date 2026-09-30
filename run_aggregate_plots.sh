#!/bin/bash
set -euo pipefail

# =============================================================================
# PLOTTING
#
# Run from the project root:
#     ./run_aggregate_plots.sh
# =============================================================================

# ─────── CONFIG ──────────────────────────────────────────────────────────────
# "all" or a space-separated subset of analysis groups.
FIGURES="all"

# "all" or a space-separated subset of experiment names.
EXPERIMENTS="all"
# OPTIONS: c2c12_dmso_09APR26 pc3_dmso_23MAR26 nrk_acid_13APR26

MOSAICS="c2c12_chambers_dff_stack c2c12_corr_pca_responses c2c12_learning_scores dmso_responder_overview c2c12_ch3_dff_pair nrk_chambers_hw_log nrk_chambers_dff_corr"

AGGREGATE_PDF=false
# ─────────────────────────────────────────────────────────────────────────────

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
