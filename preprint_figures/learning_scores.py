#!/usr/bin/env python3
"""Learning-score histograms (DMSO experiments only).

Per DMSO experiment, six figures (3 measures × 2 metrics):
    * learning_habituation_height.png  / learning_habituation_width.png
    * learning_sensitization_height.png / learning_sensitization_width.png
    * learning_anticipation_height.png  / learning_anticipation_width.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import LEARNING_STIMS_PER_TRAIN, PEAK_OFFSET
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stim_helpers import (
    compute_stim_caps,
    per_cell_response_delta,
)
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def _build_learning_inputs(state, exp_name, ch, cfg, *, metric):
    """Per-channel inputs needed by every learning-score function."""
    direction = cfg.get("response_direction", "increase")
    window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
    stim_frames = cfg["stim_frames"][ch]
    df_indexed = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df_indexed.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    if not frame_cols:
        return None
    frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
    mat = df_indexed[frame_cols].values
    n_cells, n_cols = mat.shape
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    valid_stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if not valid_stim_cols:
        return None
    caps = compute_stim_caps(valid_stim_cols, n_cols)

    def f2m(frames):
        return frames_to_min(state, exp_name, ch, frames)

    per_stim_height = np.full((len(valid_stim_cols), n_cells), np.nan)
    per_stim_width = np.full((len(valid_stim_cols), n_cells), np.nan)
    for i, (sc, cap) in enumerate(zip(valid_stim_cols, caps)):
        d, w = per_cell_response_delta(
            mat, sc, direction, window,
            return_width=True, cap_col=cap, frame_to_min_fn=f2m,
        )
        per_stim_height[i] = d
        per_stim_width[i] = w

    per_stim = per_stim_width if metric == "width" else per_stim_height
    return {
        "mat": mat,
        "stim_cols": valid_stim_cols,
        "caps": caps,
        "frame_to_min_fn": f2m,
        "per_stim": per_stim,
        "per_stim_height": per_stim_height,
        "n_cells": n_cells,
        "n_cols": n_cols,
        "direction": direction,
        "window": window,
    }


def _score_running_extremum(per_stim, *, mode, n_per_train=LEARNING_STIMS_PER_TRAIN):
    """Habituation (``mode='min'``) or sensitization (``mode='max'``)."""
    n_stims, n_cells = per_stim.shape
    n_trains = n_stims // n_per_train
    score = np.zeros(n_cells, dtype=int)
    for t in range(n_trains):
        train = per_stim[t * n_per_train:(t + 1) * n_per_train]
        if mode == "min":
            current = np.full(n_cells, np.inf)
        else:
            current = np.full(n_cells, -np.inf)
        seen_any = np.zeros(n_cells, dtype=bool)
        for s in range(n_per_train):
            r = train[s]
            valid = ~np.isnan(r)
            if mode == "min":
                beats = valid & (r < current)
            else:
                beats = valid & (r > current)
            update_score = beats & seen_any
            update_value = beats | (valid & ~seen_any)
            score += update_score.astype(int)
            current = np.where(update_value, r, current)
            seen_any = seen_any | valid
    return score


def _per_cell_typical_fluctuation(mat, stim_cols, frame_to_min_fn,
                                  n_per_train=LEARNING_STIMS_PER_TRAIN,
                                  rest_min=10.0):
    """Per-cell std of frame-to-frame deltas in non-stim periods."""
    n_cells, n_cols = mat.shape
    n_trains = len(stim_cols) // n_per_train
    non_stim = np.ones(n_cols, dtype=bool)
    all_minutes = frame_to_min_fn(np.arange(n_cols))
    for t in range(n_trains):
        first_sc = stim_cols[t * n_per_train]
        last_sc = stim_cols[t * n_per_train + n_per_train - 1]
        end_min = float(frame_to_min_fn([last_sc])[0]) + rest_min
        end_col = int(np.argmin(np.abs(all_minutes - end_min)))
        non_stim[first_sc:end_col + 1] = False
    deltas = np.diff(mat, axis=1)
    keep = non_stim[:-1] & non_stim[1:]
    if not keep.any():
        return np.full(n_cells, np.nan)
    return np.nanstd(deltas[:, keep], axis=1)


def _score_anticipation(inputs, *, metric):
    """Compute (pos_score, neg_score) per cell."""
    mat = inputs["mat"]
    stim_cols = inputs["stim_cols"]
    f2m = inputs["frame_to_min_fn"]
    direction = inputs["direction"]
    window = inputs["window"]
    n_cells, n_cols = mat.shape
    n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
    sign = -1.0 if direction == "decrease" else 1.0

    typical = _per_cell_typical_fluctuation(mat, stim_cols, f2m)

    pos = np.zeros(n_cells, dtype=int)
    neg = np.zeros(n_cells, dtype=int)
    win_lo_off, win_hi_off = window
    all_minutes = f2m(np.arange(n_cols))

    for t in range(n_trains):
        train = stim_cols[
            t * LEARNING_STIMS_PER_TRAIN:(t + 1) * LEARNING_STIMS_PER_TRAIN
        ]
        last_sc = train[-1]
        win_lo = max(0, last_sc + win_lo_off)
        win_hi = min(n_cols, last_sc + win_hi_off)
        if win_lo >= win_hi:
            continue
        seg = mat[:, win_lo:win_hi]
        if direction == "decrease":
            peak_offsets = np.argmin(
                np.where(np.isnan(seg), np.inf, seg), axis=1,
            )
        else:
            peak_offsets = np.argmax(
                np.where(np.isnan(seg), -np.inf, seg), axis=1,
            )
        peak_cols = win_lo + peak_offsets
        ref_peak_col = int(np.round(np.nanmedian(peak_cols)))
        ref_peak_min = float(f2m([ref_peak_col])[0])
        anticip_min = ref_peak_min + 10.0
        if anticip_min > float(all_minutes[-1]):
            continue
        anticip_col = int(np.argmin(np.abs(all_minutes - anticip_min)))
        if anticip_col + win_hi_off > n_cols:
            continue

        cap = compute_stim_caps([anticip_col], n_cols)[0]
        height_resp, width_resp = per_cell_response_delta(
            mat, anticip_col, direction, window,
            return_width=True, cap_col=cap, frame_to_min_fn=f2m,
        )
        signed = sign * height_resp
        significant = np.abs(height_resp) > typical
        if metric == "width":
            significant = significant & ~np.isnan(width_resp)
        pos = pos + (significant & (signed > 0)).astype(int)
        neg = neg + (significant & (signed < 0)).astype(int)

    return pos, neg


def compute_learning_scores(experiments, state):
    """Compute habituation, sensitization, anticipation scores per DMSO expt."""
    out = {}
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        out[exp_name] = {}
        for metric in ("height", "width"):
            hab_chunks, sen_chunks, pos_chunks, neg_chunks = [], [], [], []
            for ch in cfg["channels"]:
                inputs = _build_learning_inputs(
                    state, exp_name, ch, cfg, metric=metric,
                )
                if inputs is None:
                    continue
                hab = _score_running_extremum(
                    inputs["per_stim"], mode="min",
                )
                sen = _score_running_extremum(
                    inputs["per_stim"], mode="max",
                )
                pos, neg = _score_anticipation(inputs, metric=metric)
                hab_chunks.append(hab)
                sen_chunks.append(sen)
                pos_chunks.append(pos)
                neg_chunks.append(neg)
            if not hab_chunks:
                out[exp_name][metric] = None
                continue
            out[exp_name][metric] = {
                "habituation": np.concatenate(hab_chunks),
                "sensitization": np.concatenate(sen_chunks),
                "anticipation_pos": np.concatenate(pos_chunks),
                "anticipation_neg": np.concatenate(neg_chunks),
            }
            n = len(out[exp_name][metric]["habituation"])
            print(
                f"  learning scores: {exp_name} ({metric}) — {n} cells pooled "
                f"across {len(cfg['channels'])} channels."
            )
    return out


def _plot_score_histogram(scores, *, title, xlabel, save_path,
                          bins=None, color=None):
    """Single-distribution histogram (used for habituation / sensitization)."""
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    if bins is None:
        max_v = int(np.nanmax(scores)) if scores.size else 0
        bins = np.arange(-0.5, max_v + 1.5, 1)
    ax.hist(
        scores, bins=bins,
        color=color or PLOT_PARAMS["fit_color"],
        alpha=0.85, edgecolor="#222222", linewidth=0.6,
    )
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def _plot_anticipation_histogram(pos, neg, *, title, save_path):
    """Overlapping pos/neg histograms with transparent bars."""
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    max_v = int(max(np.nanmax(pos) if pos.size else 0,
                    np.nanmax(neg) if neg.size else 0))
    bins = np.arange(-0.5, max_v + 1.5, 1)
    ax.hist(
        pos, bins=bins, color="#1a9d51", alpha=0.45,
        edgecolor="#0d4f29", linewidth=0.8, label=f"positive (n={pos.size})",
    )
    ax.hist(
        neg, bins=bins, color="#e74c3c", alpha=0.45,
        edgecolor="#7a1a12", linewidth=0.8, label=f"negative (n={neg.size})",
    )
    ax.set_xlabel("Anticipation score (events / 3 trains)",
                  fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_learning_score_histograms(experiments, state, scores=None):
    """Emit habituation / sensitization / anticipation histograms (DMSO only)."""
    if scores is None:
        scores = compute_learning_scores(experiments, state)
    for exp_name, by_metric in scores.items():
        for metric, blob in by_metric.items():
            if blob is None:
                continue
            metric_label = "height" if metric == "height" else "width"
            _plot_score_histogram(
                blob["habituation"],
                title=(
                    f"{exp_name} — habituation score ({metric_label})\n"
                    f"new minimums per cell across 3 trains × 4 increments "
                    f"(max 12; first stim per train sets baseline)"
                ),
                xlabel=f"Habituation score ({metric_label})",
                save_path=fig_path(exp_name, f"learning_habituation_{metric}"),
            )
            _plot_score_histogram(
                blob["sensitization"],
                title=(
                    f"{exp_name} — sensitization score ({metric_label})\n"
                    f"new maximums per cell across 3 trains × 4 increments "
                    f"(max 12; first stim per train sets baseline)"
                ),
                xlabel=f"Sensitization score ({metric_label})",
                save_path=fig_path(exp_name, f"learning_sensitization_{metric}"),
            )
            _plot_anticipation_histogram(
                blob["anticipation_pos"],
                blob["anticipation_neg"],
                title=(
                    f"{exp_name} — anticipation score ({metric_label})\n"
                    f"|response| > typical fluctuation, 10 min after last "
                    f"train peak"
                ),
                save_path=fig_path(exp_name, f"learning_anticipation_{metric}"),
            )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_learning_score_histograms(experiments, state)


if __name__ == "__main__":
    main()
