#!/usr/bin/env python3
"""Learning-score histograms (DMSO experiments only).

Per DMSO experiment, the summed-distribution figures for habituation and
sensitization × height/width, each overlaid with a shuffled-stim-order
permutation null and the per-cell p-value count:
    * learning_habituation_height.png
    * learning_habituation_width.png
    * learning_sensitization_height.png
    * learning_sensitization_width.png
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
from common.permutation_null import permutation_null_distribution, pvalue_one_tailed
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stim_helpers import compute_stim_caps, per_cell_response_delta
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
        "per_stim": per_stim,
        "n_cells": n_cells,
        "cell_ids": list(df_indexed.index),
    }


def _score_running_extremum(per_stim, *, mode, n_per_train=LEARNING_STIMS_PER_TRAIN):
    """Habituation (``mode='min'``) or sensitization (``mode='max'``).

    Returns ``(summed, per_train)`` where ``summed.shape == (n_cells,)`` and
    ``per_train.shape == (n_trains, n_cells)``. ``summed == per_train.sum(0)``.
    """
    n_stims, n_cells = per_stim.shape
    n_trains = n_stims // n_per_train
    per_train = np.zeros((n_trains, n_cells), dtype=int)
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
            per_train[t] += update_score.astype(int)
            current = np.where(update_value, r, current)
            seen_any = seen_any | valid
    return per_train.sum(axis=0), per_train


def compute_learning_scores(experiments, state, *, n_perm=200):
    """Compute habituation + sensitization scores per DMSO experiment.

    Returns a nested dict ``out[exp][metric]`` whose blob holds the observed
    ``habituation`` / ``sensitization`` scores plus, when ``n_perm > 0``, the
    shuffled-stim-order null distributions and one-tailed per-cell p-values.
    Experiments are filtered to ``response_direction == "increase"`` (DMSO).
    """
    out = {}
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        out[exp_name] = {}
        for metric in ("height", "width"):
            hab_chunks, sen_chunks = [], []
            hab_null_chunks, sen_null_chunks = [], []
            for ch in cfg["channels"]:
                inputs = _build_learning_inputs(
                    state, exp_name, ch, cfg, metric=metric,
                )
                if inputs is None:
                    continue
                hab, _ = _score_running_extremum(inputs["per_stim"], mode="min")
                sen, _ = _score_running_extremum(inputs["per_stim"], mode="max")
                hab_chunks.append(hab)
                sen_chunks.append(sen)

                if n_perm and n_perm > 0:
                    hab_null = permutation_null_distribution(
                        lambda ps: _score_running_extremum(ps, mode="min")[0],
                        inputs["per_stim"], n_perm=n_perm,
                        rng_seed=42, mode="per_cell",
                    )
                    sen_null = permutation_null_distribution(
                        lambda ps: _score_running_extremum(ps, mode="max")[0],
                        inputs["per_stim"], n_perm=n_perm,
                        rng_seed=43, mode="per_cell",
                    )
                    hab_null_chunks.append(hab_null)
                    sen_null_chunks.append(sen_null)

            if not hab_chunks:
                out[exp_name][metric] = None
                continue
            blob = {
                "habituation": np.concatenate(hab_chunks),
                "sensitization": np.concatenate(sen_chunks),
            }
            if hab_null_chunks:
                blob["habituation_null"] = np.concatenate(hab_null_chunks, axis=1)
                blob["sensitization_null"] = np.concatenate(sen_null_chunks, axis=1)
                blob["habituation_pvalue"] = pvalue_one_tailed(
                    blob["habituation"], blob["habituation_null"],
                )
                blob["sensitization_pvalue"] = pvalue_one_tailed(
                    blob["sensitization"], blob["sensitization_null"],
                )
            out[exp_name][metric] = blob
            print(
                f"  learning scores: {exp_name} ({metric}) — "
                f"{len(blob['habituation'])} cells pooled across "
                f"{len(cfg['channels'])} channels."
            )
    return out


def _plot_score_histogram(scores, *, title, xlabel, save_path,
                          bins=None, color=None,
                          null_dist=None, pvalues=None, alpha_p=0.01):
    """Single-distribution histogram with optional permutation-null overlay.

    When ``null_dist`` (shape ``(n_perm, n_cells)``) is provided, plot the
    pooled null as a back-layer histogram normalized to the same total cell
    count, and add the count of cells with p-value < ``alpha_p`` to the
    legend (when ``pvalues`` is supplied).
    """
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    if bins is None:
        max_v = int(np.nanmax(scores)) if scores.size else 0
        if null_dist is not None and null_dist.size:
            max_v = max(max_v, int(np.nanmax(null_dist)))
        bins = np.arange(-0.5, max_v + 1.5, 1)
    if null_dist is not None and null_dist.size:
        n_perm = null_dist.shape[0]
        ax.hist(
            null_dist.ravel(), bins=bins,
            color="#9a9a9a", alpha=0.4,
            edgecolor="#555555", linewidth=0.4,
            weights=np.full(null_dist.size, 1.0 / max(n_perm, 1)),
            label=f"Shuffled null (mean over {n_perm} perms)",
            zorder=1,
        )
    ax.hist(
        scores, bins=bins,
        color=color or PLOT_PARAMS["fit_color"],
        alpha=0.85, edgecolor="#222222", linewidth=0.6,
        label=f"Observed (n={scores.size})",
        zorder=2,
    )
    if pvalues is not None and pvalues.size:
        n_sig = int(np.sum(pvalues < alpha_p))
        ax.text(
            0.98, 0.95,
            f"p<{alpha_p}: {n_sig} cells",
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"],
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
        )
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    if null_dist is not None and null_dist.size:
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_learning_score_histograms(experiments, state, scores=None):
    """Emit summed habituation / sensitization histograms (DMSO only).

    For each measure × metric: ``learning_<measure>_<metric>.png``, with the
    shuffled-stim-order null overlaid when it was computed.
    """
    if scores is None:
        scores = compute_learning_scores(experiments, state)
    for exp_name, by_metric in scores.items():
        for metric, blob in by_metric.items():
            if blob is None:
                continue
            for measure_key, label_word in (
                ("habituation", "Habituation"),
                ("sensitization", "Sensitization"),
            ):
                summed = blob[measure_key]
                null = blob.get(f"{measure_key}_null")
                pvals = blob.get(f"{measure_key}_pvalue")
                _plot_score_histogram(
                    summed,
                    title=(
                        f"{exp_name} — {label_word.lower()} score ({metric})\n"
                        f"new {'minimums' if measure_key == 'habituation' else 'maximums'} "
                        f"per cell across trains × increments"
                    ),
                    xlabel=f"{label_word} score ({metric})",
                    save_path=fig_path(
                        exp_name, f"learning_{measure_key}_{metric}",
                    ),
                    null_dist=null,
                    pvalues=pvals,
                )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    scores = compute_learning_scores(experiments, state)
    plot_learning_score_histograms(experiments, state, scores=scores)


if __name__ == "__main__":
    main()
