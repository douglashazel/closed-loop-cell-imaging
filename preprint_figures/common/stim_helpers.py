"""Stimulus-timing helpers + F0 baseline + per-cell response delta.

Copied verbatim from april28_final_figures.py.
"""

import sys

import numpy as np

from common.time_axis import frames_to_min

# io_utils lives under SCRIPTS/ at the project root.
sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def stim_spans_min(state, exp_name, ch, cfg):
    """Return ``(spans, label)`` for the channel's stimulus shaded blocks."""
    stim_frames = cfg["stim_frames"][ch]
    if not stim_frames:
        return [], cfg.get("stim_label", "Stimulus")
    duration = float(cfg.get("stim_duration_minutes", 0.0) or 0.0)
    starts = frames_to_min(state, exp_name, ch, stim_frames)
    spans = [(float(s), float(s) + duration) for s in starts]
    return spans, cfg.get("stim_label", "Stimulus")


def stim_timing_aligned_across_channels(cfg, state, exp_name, tol_min=0.5):
    """True if every channel's stim minutes match the first channel's."""
    channels = cfg.get("channels", [])
    if len(channels) <= 1:
        return True
    sf_dict = cfg.get("stim_frames")
    if not isinstance(sf_dict, dict):
        return True
    ref_ch = channels[0]
    ref_frames = sf_dict.get(ref_ch, [])
    if not ref_frames:
        return True
    ref_min = np.asarray(frames_to_min(state, exp_name, ref_ch, ref_frames))
    for ch in channels[1:]:
        sf = sf_dict.get(ch, [])
        if len(sf) != len(ref_frames):
            return False
        ch_min = np.asarray(frames_to_min(state, exp_name, ch, sf))
        if np.any(np.abs(ch_min - ref_min) > tol_min):
            return False
    return True


def draw_stim_spans(ax, spans, label, color, alpha=0.18):
    """Shade each ``(start, end)`` span on ``ax``; label only the first."""
    for idx, (start_m, end_m) in enumerate(spans):
        ax.axvspan(
            start_m, end_m,
            color=color, alpha=alpha,
            linewidth=0, zorder=0,
            label=label if idx == 0 else None,
        )


def compute_stim_caps(stim_cols, n_cols, *, uniform_cap_cols=None):
    """Return per-stim "cap column" for the width search.

    Cap policy:
        * If ``uniform_cap_cols`` is given, every stim caps at
          ``stim_col + uniform_cap_cols`` (clamped to ``n_cols - 1``).
          Use this to make widths comparable across pulses with uneven
          inter-stim spacing.
        * Otherwise: stim *i* (not last) caps at ``stim_cols[i + 1]`` and
          the very last stim caps at ``n_cols - 1``.
    """
    if uniform_cap_cols is not None:
        return [
            int(min(int(sc) + int(uniform_cap_cols), n_cols - 1))
            for sc in stim_cols
        ]
    out = []
    for i, sc in enumerate(stim_cols):
        if i + 1 < len(stim_cols):
            out.append(int(stim_cols[i + 1]))
        else:
            out.append(int(n_cols - 1))
    return out


def per_cell_response_delta(
    values_by_col, stim_col, direction, window,
    *,
    return_width=False, cap_col=None, frame_to_min_fn=None,
):
    """Return per-cell ``response_value − baseline`` for one stimulus.

    See april28_final_figures.py for full doc.
    """
    n_cells, n_cols = values_by_col.shape
    lo, hi = window
    if stim_col < 0 or stim_col >= n_cols:
        empty = np.full(n_cells, np.nan)
        return (empty, empty.copy()) if return_width else empty
    base = values_by_col[:, stim_col]
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
            "per_cell_response_delta(return_width=True) requires both "
            "`cap_col` and `frame_to_min_fn`."
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


def compute_f0_baseline(state, exp_name, ch, cfg):
    """Per-cell F0 = mean of corrected luminosity from frame 0 up to first stim.

    Returns
    -------
    F0 : np.ndarray of shape (n_cells, 1)
    baseline_cols : list[str]
    first_stim : int
    """
    stim_frames = cfg.get("stim_frames", {}).get(ch, [])
    if len(stim_frames):
        first_stim = int(min(stim_frames))
    else:
        first_stim = 1

    df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
    frame_cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    baseline_cols = [
        c for c in frame_cols if int(str(c).lstrip("f")) < first_stim
    ]
    if not baseline_cols:
        baseline_cols = frame_cols[:1]
    F0 = np.nanmean(df[baseline_cols].values, axis=1, keepdims=True)
    return F0, baseline_cols, first_stim
