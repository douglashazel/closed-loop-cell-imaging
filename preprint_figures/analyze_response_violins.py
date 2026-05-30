#!/usr/bin/env python3
"""Per-stimulus response-violin analysis (no plotting) → analysis_cache/<exp>/response_violins.pkl.

Computes the figure-ready intermediates the response-violin figures display, for
each DMSO experiment and each metric in {"height", "width"} (signal "dff"):

  * per-stim per-cell pooled arrays — ``violin_data`` (ragged list of 1-D arrays,
    NaN-dropped per stim) and the parallel ``responder_data`` (ragged bool arrays),
  * the per-stim x labels (stim onset minutes),
  * per-train cell means + train-level stats (within-experiment Friedman +
    replicate-level one-sample t across channels → ``train_p`` / ``stats_text``),
  * ``chan_train_means`` (n_channels × n_trains) for the per-replicate figure.

The matplotlib rendering lives in ``plots/response_violins.py``; this script
writes only numbers. The response math is the verbatim
``per_cell_response_delta_with_baseline`` port from the original
``response_violins.py`` — identical baseline window, width caps, and stats.
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_responders import get_responder_masks
from common.baseline_window import (
    per_cell_response_delta_with_baseline,
    prestim_baseline_values,
)
from common.cli import parse_args
from common.config import LEARNING_STIMS_PER_TRAIN, PEAK_OFFSET, cell_line_label
from common.io_paths import save_analysis_cache
from common.pipeline import prepare_state
from common.stats import friedman_with_posthoc, inferential_caveat, one_sample_t_dz
from common.stim_helpers import compute_f0_baseline, compute_stim_caps
from common.time_axis import frames_to_min, response_window_frames

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


# Pre-stim baseline window used for every per-stim Δ — matches the responder
# gate in common/responders.py so the descriptive figures and the responder
# classification share the same baseline definition.
_PRESTIM_BASELINE_FRAMES = 5

METRICS = ("height", "width")


# =============================================================================
# Per-channel response arrays — verbatim port from response_violins.py.
# =============================================================================
def _build_signal_matrix(state, exp_name, ch, cfg, *, signal):
    """Return ``(values_by_col, df_indexed, frame_cols, frame_to_col)`` for the chosen signal.

    ``signal == "lum"`` returns the raw corrected luminosity matrix; ``"dff"``
    returns ``(mat - F0) / F0`` so amplitude and width violins can be plotted
    in normalized units.
    """
    df_indexed = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df_indexed.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
    mat = df_indexed[frame_cols].values
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    if signal == "dff":
        F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
        F0_safe = np.where(F0 == 0, np.nan, F0)
        values = (mat - F0) / F0_safe
    else:
        values = mat
    return values, df_indexed, frame_cols, frame_to_col


def _per_channel_stim_response_arrays(
    state, exp_name, ch, cfg, *, metric, signal="lum", responder_mask=None,
):
    """Return ``(violin_data, responder_data, base_min, n_total)`` for one (exp, channel).

    ``signal`` selects between raw luminosity (default) and dF/F0; both produce
    the same return shapes so the caller can swap freely.

    ``responder_data`` mirrors ``violin_data`` — for each stimulus a boolean
    array marking which of the (NaN-dropped) per-cell values belong to a
    responder cell. When ``responder_mask`` is None every entry is False.
    """
    direction = cfg.get("response_direction", "increase")
    window = response_window_frames(state, exp_name, ch, cfg)

    stim_frames = cfg["stim_frames"][ch]
    values, df_indexed, frame_cols, frame_to_col = _build_signal_matrix(
        state, exp_name, ch, cfg, signal=signal,
    )
    n_cells, n_cols = values.shape

    valid_stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if metric == "width" and valid_stim_cols:
        if len(valid_stim_cols) >= 2:
            uniform_cap = int(np.min(np.diff(valid_stim_cols)))
        else:
            uniform_cap = max(0, n_cols - 1 - valid_stim_cols[0])
        caps = compute_stim_caps(
            valid_stim_cols, n_cols, uniform_cap_cols=uniform_cap,
        )
    else:
        caps = []
    cap_for_col = dict(zip(valid_stim_cols, caps))

    def _f2m(frames):
        return frames_to_min(state, exp_name, ch, frames)

    violin_data = []
    responder_data = []
    for p in stim_frames:
        if p not in frame_to_col:
            violin_data.append(np.array([]))
            responder_data.append(np.array([], dtype=bool))
            continue
        col = frame_to_col[p]
        baseline = prestim_baseline_values(
            values, col, n_pre=_PRESTIM_BASELINE_FRAMES,
        )
        if metric == "width":
            _, widths = per_cell_response_delta_with_baseline(
                values, col, direction, window, baseline,
                return_width=True,
                cap_col=cap_for_col[col],
                frame_to_min_fn=_f2m,
            )
            finite = ~np.isnan(widths)
            vals = widths[finite]
        else:
            deltas = per_cell_response_delta_with_baseline(
                values, col, direction, window, baseline,
            )
            finite = ~np.isnan(deltas)
            vals = deltas[finite]
        violin_data.append(vals)
        if responder_mask is not None:
            responder_data.append(np.asarray(responder_mask, dtype=bool)[finite])
        else:
            responder_data.append(np.zeros(vals.shape, dtype=bool))

    base_min = list(frames_to_min(state, exp_name, ch, stim_frames)) if stim_frames else []
    return violin_data, responder_data, base_min, n_cells


def _per_train_cell_means(state, exp_name, ch, cfg, *, metric, signal,
                          n_per_train=LEARNING_STIMS_PER_TRAIN):
    """Per-cell mean response within each stimulus train for one channel.

    Returns ``(cell_train_means, n_trains)`` where ``cell_train_means`` has
    shape ``(n_cells, n_trains)`` — entry (c, t) is cell c's NaN-mean response
    over the ``n_per_train`` pulses of train t. Returns ``(None, 0)`` when the
    channel's stimuli do not divide into at least two whole trains (e.g. the
    acid experiment, whose pulses are not organized into fixed trains).
    """
    direction = cfg.get("response_direction", "increase")
    window = response_window_frames(state, exp_name, ch, cfg)
    stim_frames = cfg["stim_frames"][ch]
    n_stims = len(stim_frames)
    n_trains = n_stims // n_per_train
    if n_stims == 0 or n_stims % n_per_train != 0 or n_trains < 2:
        return None, 0

    values, _, _, frame_to_col = _build_signal_matrix(
        state, exp_name, ch, cfg, signal=signal,
    )
    n_cells, n_cols = values.shape

    valid_stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if metric == "width" and valid_stim_cols:
        if len(valid_stim_cols) >= 2:
            uniform_cap = int(np.min(np.diff(valid_stim_cols)))
        else:
            uniform_cap = max(0, n_cols - 1 - valid_stim_cols[0])
        caps = compute_stim_caps(
            valid_stim_cols, n_cols, uniform_cap_cols=uniform_cap,
        )
    else:
        caps = []
    cap_for_col = dict(zip(valid_stim_cols, caps))

    def _f2m(frames):
        return frames_to_min(state, exp_name, ch, frames)

    # Per-cell response per stim, keeping cell alignment and NaNs (a missing
    # stim leaves an all-NaN row; nanmean over the train then ignores it).
    per_stim = np.full((n_stims, n_cells), np.nan)
    for i, p in enumerate(stim_frames):
        if p not in frame_to_col:
            continue
        col = frame_to_col[p]
        baseline = prestim_baseline_values(
            values, col, n_pre=_PRESTIM_BASELINE_FRAMES,
        )
        if metric == "width":
            _, widths = per_cell_response_delta_with_baseline(
                values, col, direction, window, baseline,
                return_width=True, cap_col=cap_for_col[col],
                frame_to_min_fn=_f2m,
            )
            per_stim[i] = widths
        else:
            per_stim[i] = per_cell_response_delta_with_baseline(
                values, col, direction, window, baseline,
            )

    trains = per_stim.reshape(n_trains, n_per_train, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN train
        cell_train_means = np.nanmean(trains, axis=1).T  # (n_cells, n_trains)
    return cell_train_means, n_trains


def _train_level_stats(per_channel_means, *, metric, n_trains):
    """Cell-level Friedman + replicate-level one-sample t across stim trains.

    ``per_channel_means`` is a list of ``(n_cells, n_trains)`` arrays (or None),
    one per channel — channels are the biological replicates. Two layers are
    reported: a within-experiment Friedman repeated-measures test treating
    cells as observations (with Holm-corrected Wilcoxon post-hoc per train
    pair), and a replicate-level one-sample t-test of the per-channel
    last-minus-first train difference against 0. Returns
    ``(stats_text, chan_train_means, train_p)`` — ``chan_train_means`` has
    shape ``(n_channels, n_trains)`` for the inset and ``train_p`` is the
    replicate-level p-value for that difference — or ``(None, None, None)``.
    """
    arrays = [a for a in per_channel_means if a is not None]
    if not arrays or n_trains < 2:
        return None, None, None

    pooled = np.concatenate(arrays, axis=0)  # (total_cells, n_trains)
    n_total = pooled.shape[0]
    complete = pooled[~np.isnan(pooled).any(axis=1)]
    n_complete = complete.shape[0]

    lines = [f"Train-level comparison ({metric};  T1 … T{n_trains})"]

    # Cell-level layer — within-experiment, cells are NOT biological replicates.
    fr = friedman_with_posthoc(complete)
    if fr.get("insufficient"):
        lines.append(
            f"cell-level: only {n_complete} cells tracked across all "
            f"trains — too few for Friedman"
        )
    else:
        lines.append(
            f"cell-level (within-expt; {n_complete}/{n_total} cells tracked "
            f"across all trains):"
        )
        lines.append(
            f"  Friedman chi2={fr['friedman_stat']:.1f}, "
            f"p={fr['friedman_p']:.3g}, Kendall W={fr['kendall_w']:.3f}"
        )
        for ph in fr["posthoc"]:
            i, j = ph["pair"]
            lines.append(
                f"  T{i + 1}-T{j + 1}: Wilcoxon p={ph['p_adj']:.3g} (Holm), "
                f"rank-biserial r={ph['rank_biserial']:+.3f}"
            )

    # Replicate-level layer — channels are the biological replicates.
    chan_means = np.vstack([np.nanmean(a, axis=0) for a in arrays])
    n_ch = chan_means.shape[0]
    diffs = chan_means[:, -1] - chan_means[:, 0]
    train_p = None
    if n_ch >= 2:
        t = one_sample_t_dz(diffs)
        train_p = t["p_value"]
        per_ch = "  ".join(f"{d:+.3f}" for d in diffs)
        lines.append(
            f"replicate-level ({n_ch} channels) "
            f"deltaT{n_trains}-T1 per replicate = [{per_ch}]"
        )
        lines.append(
            f"  1-sample t vs 0: p={t['p_value']:.3g}, "
            f"Cohen dz={t['cohen_dz']:+.2f}"
        )
    else:
        lines.append(
            "replicate-level: 1 channel — within-experiment only, "
            "no biological-replicate test"
        )
    # The overlay shows across-replicate spread; with a single channel there
    # are no replicate points to plot, so suppress it (the lone line would
    # just be the pooled per-train cell mean, not a replicate mean).
    return "\n".join(lines), (chan_means if n_ch >= 2 else None), train_p


# =============================================================================
# Per-experiment, per-metric assembly — verbatim port of the source loop body,
# minus the matplotlib draw calls.
# =============================================================================
def _metric_bundle(experiments, state, exp_name, cfg, *, metric, signal,
                   responder_masks):
    """Compute one metric's full cache bundle for one experiment.

    Returns ``None`` when the experiment's channels have differing stim counts
    (or no stims), matching the source ``continue`` skip.
    """
    signal_label = "dF/F₀" if signal == "dff" else "Δ fluorescence"
    direction = cfg.get("response_direction", "increase")
    extremum_label = "max" if direction == "increase" else "min"
    win_min = cfg.get("response_window_minutes")
    if win_min is not None:
        window_str = f"{win_min[0]:g}–{win_min[1]:g} min post-stim"
    else:
        wf = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        window_str = f"stim+{wf[0]}…stim+{wf[1] - 1} frames"
    if metric == "width":
        unit_str = "(dF/F₀ baseline)" if signal == "dff" else ""
        y_label = f"Response width (min) {unit_str}".strip()
        width_cap_note = " — width search capped at min inter-stim interval"
    else:
        y_label = f"{signal_label}  ({extremum_label} − baseline)"
        width_cap_note = ""

    channels = cfg["channels"]
    stim_counts = [len(cfg["stim_frames"][ch]) for ch in channels]
    if not channels or len(set(stim_counts)) != 1 or stim_counts[0] == 0:
        print(
            f"{exp_name}: pooled response violin ({metric}) — channels "
            f"have differing stim counts {stim_counts} or no stims; skipping."
        )
        return None
    n_stims = stim_counts[0]
    pooled = [[] for _ in range(n_stims)]
    pooled_resp = [[] for _ in range(n_stims)]
    ref_base_min = None
    total_cells = 0
    total_responders = 0
    per_channel_train_means = []
    train_n = 0
    for ch in channels:
        ch_mask = responder_masks.get((exp_name, ch)) if responder_masks else None
        vd, rd, base_min, n_cells = _per_channel_stim_response_arrays(
            state, exp_name, ch, cfg, metric=metric, signal=signal,
            responder_mask=ch_mask,
        )
        ctm, n_tr = _per_train_cell_means(
            state, exp_name, ch, cfg, metric=metric, signal=signal,
        )
        per_channel_train_means.append(ctm)
        train_n = max(train_n, n_tr)
        if ref_base_min is None:
            ref_base_min = base_min
        total_cells += n_cells
        if ch_mask is not None:
            total_responders += int(np.asarray(ch_mask, dtype=bool).sum())
        for i, arr in enumerate(vd):
            if arr.size:
                pooled[i].append(arr)
                pooled_resp[i].append(rd[i])
    violin_data = [
        np.concatenate(p) if p else np.array([]) for p in pooled
    ]
    responder_data = [
        np.concatenate(p) if p else np.array([], dtype=bool)
        for p in pooled_resp
    ]
    x_labels = [f"{bm:.1f}" for bm in (ref_base_min or [])]
    n_complete = sum(len(v) > 0 for v in violin_data)
    stats_text, chan_means, train_p = _train_level_stats(
        per_channel_train_means, metric=metric, n_trains=train_n,
    )
    caveat = inferential_caveat(exp_name, len(channels), unit="cell")
    print(
        f"{exp_name}: pooled response violin ({metric}) — "
        f"{n_complete}/{n_stims} stims have data, {total_cells} cells "
        f"across {len(channels)} channels."
    )
    if responder_masks is not None:
        print(
            f"{exp_name}: pooled response violin ({metric}, responders) — "
            f"{total_responders}/{total_cells} cells flagged as responders."
        )

    # The "{extremum} over {window_str} − baseline; N cells, M channels" core
    # the violin title interpolates (preserved verbatim, with {placeholders}
    # the spec fills from this cache).
    title_core = (
        f"{extremum_label} over {window_str} − baseline; "
        f"{total_cells} cells, {len(channels)} channels"
    )

    return {
        "violin_data": violin_data,
        "responder_data": responder_data,
        "x_labels": x_labels,
        "chan_train_means": (None if chan_means is None else np.asarray(chan_means, dtype=float)),
        "train_p": (None if train_p is None else float(train_p)),
        "stats_text": stats_text,
        "n_total_cells": int(total_cells),
        "n_channels": int(len(channels)),
        "n_total_responders": int(total_responders),
        "y_label": y_label,
        "title_core": title_core,
        "width_cap_note": width_cap_note,
        "caveat": caveat,
    }


def analyze(experiments, state, *, signal="dff"):
    """Compute + cache one ``response_violins`` bundle per DMSO experiment.

    Only experiments whose pulses divide into fixed trains carry train-level
    stats (the acid experiment yields ``chan_train_means=None`` / ``train_p=None``
    — same as the source). The bundle holds both metrics under ``["height"]`` /
    ``["width"]``.
    """
    responder_masks = get_responder_masks(experiments, state)
    for exp_name, cfg in experiments.items():
        data = {}
        for metric in METRICS:
            bundle = _metric_bundle(
                experiments, state, exp_name, cfg,
                metric=metric, signal=signal, responder_masks=responder_masks,
            )
            if bundle is not None:
                data[metric] = bundle
        if not data:
            print(f"  {exp_name}: no response-violin data — skipping cache")
            continue
        meta = {
            "exp_name": exp_name,
            "cell_line": cell_line_label(exp_name),
            "caveat": inferential_caveat(exp_name, len(cfg["channels"]), unit="cell"),
        }
        save_analysis_cache(data, exp_name, "response_violins", meta=meta)
        print(
            f"  cached response_violins.pkl for {exp_name} "
            f"({len(data)} metrics)"
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
