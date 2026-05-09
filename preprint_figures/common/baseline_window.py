"""Alternative baseline definitions for response amplitude/width.

The default ``per_cell_response_delta`` uses the value at the stim frame as
both the delta reference and the width-crossing threshold. For NRK acid
experiments two additional baselines make sense:

* **prestim_window** — mean of N frames immediately before the stim, more
  robust to instantaneous noise at the stim frame.
* **nrk_setpoint** — the hardware feedback setpoint active at the stim
  frame. Width measures how long the cell stays below the setpoint.
"""

import json
import os

import numpy as np

from common.time_axis import setpoint_regions_from_log


_NRK_LOG_CACHE = {}


def prestim_baseline_values(values_by_col, stim_col, *, n_pre=5):
    """Per-cell mean over ``[stim_col - n_pre, stim_col)``.

    Falls back to the stim-frame value when the pre-stim window is empty
    (e.g. ``stim_col == 0``).
    """
    n_cells, n_cols = values_by_col.shape
    lo = max(0, int(stim_col) - int(n_pre))
    hi = int(stim_col)
    if lo >= hi:
        return values_by_col[:, max(0, int(stim_col))].astype(float)
    return np.nanmean(values_by_col[:, lo:hi], axis=1)


def _load_nrk_log_entries(cfg, ch):
    """Read and cache the per-channel luminosity log."""
    key = (id(cfg), ch)
    if key in _NRK_LOG_CACHE:
        return _NRK_LOG_CACHE[key]
    log_path, ch_num = cfg["stim_logs"][ch]
    lum_log_path = os.path.join(
        os.path.dirname(log_path),
        f"luminosity_log_channel{ch_num}.json",
    )
    with open(lum_log_path) as f:
        entries = json.load(f)
    entries = [e for e in entries if e.get("channel") == ch_num]
    entries.sort(key=lambda e: e["frame"])
    _NRK_LOG_CACHE[key] = entries
    return entries


def nrk_setpoint_at_frame(experiments, exp_name, ch, frame_idx):
    """Return the NRK setpoint value active at ``frame_idx``.

    Returns ``None`` for non-NRK channels, missing logs, or frames outside
    every parsed region.
    """
    cfg = experiments.get(exp_name)
    if cfg is None or "stim_logs" not in cfg or ch not in cfg["stim_logs"]:
        return None
    try:
        entries = _load_nrk_log_entries(cfg, ch)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    regions = setpoint_regions_from_log(entries)
    if not regions:
        return None
    for start, end, sp in regions:
        if start <= frame_idx <= end:
            return float(sp)
    nearest = min(
        regions,
        key=lambda r: min(abs(r[0] - frame_idx), abs(r[1] - frame_idx)),
    )
    return float(nearest[2])


def per_cell_response_delta_with_baseline(
    values_by_col, stim_col, direction, window, baseline,
    *, return_width=False, cap_col=None, frame_to_min_fn=None,
):
    """Variant of ``stim_helpers.per_cell_response_delta`` with an explicit baseline.

    ``baseline`` may be a scalar (broadcast to all cells) or a per-cell array
    of shape ``(n_cells,)``. Both delta and width crossings use this baseline.
    """
    n_cells, n_cols = values_by_col.shape
    lo, hi = window
    if stim_col < 0 or stim_col >= n_cols:
        empty = np.full(n_cells, np.nan)
        return (empty, empty.copy()) if return_width else empty
    base = np.broadcast_to(np.asarray(baseline, dtype=float), (n_cells,)).astype(float)
    win_lo = max(0, stim_col + lo)
    win_hi = min(n_cols, stim_col + hi)
    if win_lo >= win_hi:
        empty = np.full(n_cells, np.nan)
        return (empty, empty.copy()) if return_width else empty
    win = values_by_col[:, win_lo:win_hi]
    if direction == "decrease":
        extremum = np.nanmin(win, axis=1)
        peak_offsets = np.nanargmin(
            np.where(np.isnan(win), np.inf, win), axis=1,
        )
    else:
        extremum = np.nanmax(win, axis=1)
        peak_offsets = np.nanargmax(
            np.where(np.isnan(win), -np.inf, win), axis=1,
        )
    deltas = extremum - base

    if not return_width:
        return deltas

    if cap_col is None or frame_to_min_fn is None:
        raise ValueError(
            "per_cell_response_delta_with_baseline(return_width=True) requires "
            "both `cap_col` and `frame_to_min_fn`."
        )
    cap_col = int(min(max(cap_col, win_lo), n_cols - 1))
    base_min = float(frame_to_min_fn([stim_col])[0])
    cap_min = float(frame_to_min_fn([cap_col])[0])

    widths = np.full(n_cells, np.nan, dtype=np.float64)
    peak_cols = win_lo + peak_offsets
    for i in range(n_cells):
        b = base[i]
        if np.isnan(deltas[i]) or np.isnan(b):
            continue
        pc = int(peak_cols[i])
        scan_lo = pc + 1
        scan_hi = cap_col + 1
        if scan_lo >= scan_hi:
            widths[i] = max(0.0, cap_min - base_min)
            continue
        seg = values_by_col[i, scan_lo:scan_hi]
        if direction == "decrease":
            crossings = np.where(seg >= b)[0]
        else:
            crossings = np.where(seg <= b)[0]
        if crossings.size:
            cross_col = scan_lo + int(crossings[0])
            cross_min = float(frame_to_min_fn([cross_col])[0])
            widths[i] = max(0.0, cross_min - base_min)
        else:
            widths[i] = max(0.0, cap_min - base_min)

    return deltas, widths
