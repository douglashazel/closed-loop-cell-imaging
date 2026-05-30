#!/usr/bin/env python3
"""dF/F0 analysis (no plotting) → analysis_cache/<exp>/dff.pkl.

Computes the figure-ready intermediates the dF/F0 figures display:
  * per (experiment, channel): corrected-fluorescence matrix, dF/F0 matrix,
    time axis, F0 note, stim spans, responder mask, real-setpoint marker
  * per experiment: pooled-across-channels dF/F0 (all cells; responders only,
    with mean ± SEM)

The matplotlib rendering lives in ``plots/dff.py``; this script writes only
numbers. dF/F0 math is the verbatim ``(mat - F0) / F0_safe`` with F0 from
``compute_f0_baseline`` — identical to the original ``dff.py``.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_responders import get_responder_masks
from common.cli import parse_args
from common.config import cell_line_label
from common.io_paths import save_analysis_cache
from common.pipeline import prepare_state
from common.stim_helpers import (
    compute_f0_baseline,
    stim_spans_min,
    stim_timing_aligned_across_channels,
)
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def _frame_cols(df):
    """Sorted frame columns ('f0','f1',...) and their integer indices."""
    cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    nums = np.array([int(str(c).lstrip("f")) for c in cols])
    return cols, nums


def _channel_dff(state, exp_name, ch, cfg):
    """Return the per-channel dict the dff figures need."""
    df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
    frame_cols, frame_nums = _frame_cols(df)
    frame_min = frames_to_min(state, exp_name, ch, frame_nums)
    mat = df[frame_cols].values

    F0, _, first_stim = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff_mat = (mat - F0) / F0_safe

    spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
    rsp = state["real_setpoint_min"][exp_name].get(ch)
    f0_note = (
        f"F₀ = mean of baseline (frames 0–{first_stim - 1})"
        if first_stim > 1
        else "F₀ = frame 0"
    )
    return {
        "frame_min": np.asarray(frame_min, dtype=float),
        "raw_mat": np.asarray(mat, dtype=float),
        "dff_mat": np.asarray(dff_mat, dtype=float),
        "cell_ids": list(df.index),
        "first_stim": int(first_stim),
        "f0_note": f0_note,
        "spans": spans,
        "stim_label": stim_label,
        "real_setpoint_min": (None if rsp is None else float(rsp)),
        "n_stims": int(len(cfg["stim_frames"][ch])),
        "n_all": int(mat.shape[0]),
    }


def _pooled(state, exp_name, cfg, responder_masks):
    """Pool dF/F0 across channels: all cells (mean) + responders (mean ± SEM)."""
    channels = cfg["channels"]
    per_ch = {}
    for ch in channels:
        df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
        cols, _ = _frame_cols(df)
        per_ch[ch] = (df, cols)
    if any(not cols for _, cols in per_ch.values()):
        return None
    n_common = min(len(cols) for _, cols in per_ch.values())

    ref_ch = channels[0]
    ref_cols = per_ch[ref_ch][1][:n_common]
    ref_nums = np.array([int(str(c).lstrip("f")) for c in ref_cols])
    frame_min = frames_to_min(state, exp_name, ref_ch, ref_nums)

    all_rows, resp_rows, counts = [], [], []
    for ch in channels:
        df, cols = per_ch[ch]
        mat = df[cols[:n_common]].values
        F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
        F0_safe = np.where(F0 == 0, np.nan, F0)
        dff_mat = (mat - F0) / F0_safe
        all_rows.append(dff_mat)
        counts.append((ch, int(mat.shape[0])))
        mask = responder_masks.get((exp_name, ch))
        if mask is not None and np.asarray(mask, dtype=bool).any():
            resp_rows.append(dff_mat[np.asarray(mask, dtype=bool)])

    pooled_all = np.vstack(all_rows)
    out = {
        "frame_min": np.asarray(frame_min, dtype=float),
        "pooled_dff_all": pooled_all,
        "mean_all": np.nanmean(pooled_all, axis=0),
        "per_channel_counts": counts,
        "stim_aligned": bool(
            stim_timing_aligned_across_channels(cfg, state, exp_name)
        ),
        "spans": stim_spans_min(state, exp_name, ref_ch, cfg)[0],
        "stim_label": stim_spans_min(state, exp_name, ref_ch, cfg)[1],
    }
    if resp_rows:
        pooled_resp = np.vstack(resp_rows)
        n_per_col = np.sum(~np.isnan(pooled_resp), axis=0).astype(float)
        out["pooled_dff_responders"] = pooled_resp
        out["mean_responders"] = np.nanmean(pooled_resp, axis=0)
        out["sem_responders"] = (
            np.nanstd(pooled_resp, axis=0) / np.sqrt(np.maximum(n_per_col, 1))
        )
        out["n_total_responders"] = int(pooled_resp.shape[0])
    else:
        out["pooled_dff_responders"] = None
        out["mean_responders"] = None
        out["sem_responders"] = None
        out["n_total_responders"] = 0
    return out


def analyze(experiments, state):
    responder_masks = get_responder_masks(experiments, state)
    for exp_name, cfg in experiments.items():
        per_channel = {}
        for ch in cfg["channels"]:
            d = _channel_dff(state, exp_name, ch, cfg)
            mask = responder_masks.get((exp_name, ch))
            d["mask"] = (
                np.zeros(d["n_all"], dtype=bool)
                if mask is None
                else np.asarray(mask, dtype=bool)
            )
            per_channel[ch] = d
        data = {"per_channel": per_channel,
                "pooled": _pooled(state, exp_name, cfg, responder_masks)}
        meta = {"cell_line": cell_line_label(exp_name), "exp_name": exp_name}
        save_analysis_cache(data, exp_name, "dff", meta=meta)
        print(f"  cached dff.pkl for {exp_name} ({len(per_channel)} channels)")


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
