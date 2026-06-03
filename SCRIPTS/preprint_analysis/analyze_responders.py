#!/usr/bin/env python3
"""Shared responder threshold + mask analysis (no plotting).

Responders are scored by the aggregate per-stim Δ dF/F0 across all stimuli (see
``common/responders.py``). This step computes the per-(experiment, channel)
threshold and boolean mask ONCE and caches them to
``analysis_cache/<exp>/responders.pkl`` so every downstream analysis reads the
same deterministic result (``rng_seed=42``) instead of recomputing it.

Run first in ``run_aggregate_results.sh``; the consumers (dff, average_peak,
correlation_distance, response_violins) call :func:`get_responder_masks`, which
loads the cache when present and computes+caches it otherwise — so each
``analyze_*.py`` stays independently runnable.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import cell_line_label
from common.io_paths import load_analysis_cache, save_analysis_cache
from common.pipeline import prepare_state
from common.responders import (
    compute_responder_masks,
    compute_responder_thresholds,
)

sys.path.insert(0, "SCRIPTS/core_pipeline")
from io_utils import lum_dict_to_df  # noqa: E402

# Responder scoring parameters (the defaults compute_responder_* already use).
RESPONDER_PARAMS = {
    "alpha": 0.01,
    "baseline_n_pre": 5,
    "stat": "mean",
    "rng_seed": 42,
}


def _channel_cell_ids(state, exp_name, ch):
    """CellID order of the corrected-luminosity matrix (mask alignment)."""
    df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
    return list(df.index)


def compute_and_cache(experiments, state):
    """Compute thresholds + masks for ``experiments`` and cache one bundle per
    experiment. Returns ``{(exp_name, ch): bool ndarray}``."""
    thresholds = compute_responder_thresholds(
        experiments, state,
        alpha=RESPONDER_PARAMS["alpha"],
        baseline_n_pre=RESPONDER_PARAMS["baseline_n_pre"],
        stat=RESPONDER_PARAMS["stat"],
        rng_seed=RESPONDER_PARAMS["rng_seed"],
    )
    masks = compute_responder_masks(
        experiments, state, thresholds=thresholds,
        alpha=RESPONDER_PARAMS["alpha"],
        baseline_n_pre=RESPONDER_PARAMS["baseline_n_pre"],
        stat=RESPONDER_PARAMS["stat"],
    )

    for exp_name, cfg in experiments.items():
        data = {"thresholds": {}, "masks": {}, "cell_ids": {},
                "params": dict(RESPONDER_PARAMS)}
        meta = {"cell_line": cell_line_label(exp_name),
                "n_responders": {}, "n_cells": {}, "pct_resp": {}}
        for ch in cfg["channels"]:
            mask = masks.get((exp_name, ch))
            mask = np.zeros(0, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
            data["masks"][ch] = mask
            data["thresholds"][ch] = thresholds.get((exp_name, ch))
            data["cell_ids"][ch] = _channel_cell_ids(state, exp_name, ch)
            n_cells = int(mask.size)
            n_resp = int(mask.sum())
            meta["n_cells"][ch] = n_cells
            meta["n_responders"][ch] = n_resp
            meta["pct_resp"][ch] = (n_resp / n_cells) if n_cells else 0.0
        save_analysis_cache(data, exp_name, "responders", meta=meta)
        print(f"  cached responders.pkl for {exp_name} "
              f"({sum(meta['n_responders'].values())} responders / "
              f"{sum(meta['n_cells'].values())} cells)")

    return masks


def get_responder_masks(experiments, state):
    """Return ``{(exp_name, ch): bool ndarray}`` for every channel in
    ``experiments``, loading cached masks where available and computing+caching
    any experiment that is missing (so callers need no run ordering)."""
    out = {}
    missing = {}
    for exp_name, cfg in experiments.items():
        try:
            blob = load_analysis_cache(exp_name, "responders")
        except FileNotFoundError:
            missing[exp_name] = cfg
            continue
        cached = blob["data"]["masks"]
        for ch in cfg["channels"]:
            m = cached.get(ch)
            out[(exp_name, ch)] = None if m is None else np.asarray(m, dtype=bool)

    if missing:
        out.update(compute_and_cache(missing, state))
    return out


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    compute_and_cache(experiments, state)


if __name__ == "__main__":
    main()
