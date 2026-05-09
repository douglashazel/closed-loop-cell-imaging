#!/usr/bin/env python3
"""Learning-score histograms (DMSO experiments only).

Per DMSO experiment, the canonical six summed-distribution figures
(habituation/sensitization/anticipation × height/width). Each summed figure
is augmented with:
    * ``..._per_train.png``  — one panel per train showing the train-level score.
    * ``..._split.png``      — responder vs non-responder distributions overlaid.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.cluster_labels import align_labels_to_cells, load_cluster_labels
from common.config import LEARNING_STIMS_PER_TRAIN, PEAK_OFFSET
from common.permutation_null import (
    permutation_null_distribution,
    pvalue_one_tailed,
    shuffle_per_stim,
)
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_thresholds
from common.stim_helpers import (
    compute_f0_baseline,
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
    """Compute anticipation scores. Returns ``(pos, neg, pos_per_train, neg_per_train)``."""
    mat = inputs["mat"]
    stim_cols = inputs["stim_cols"]
    f2m = inputs["frame_to_min_fn"]
    direction = inputs["direction"]
    window = inputs["window"]
    n_cells, n_cols = mat.shape
    n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
    sign = -1.0 if direction == "decrease" else 1.0

    typical = _per_cell_typical_fluctuation(mat, stim_cols, f2m)

    pos_per_train = np.zeros((n_trains, n_cells), dtype=int)
    neg_per_train = np.zeros((n_trains, n_cells), dtype=int)
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
        pos_per_train[t] = (significant & (signed > 0)).astype(int)
        neg_per_train[t] = (significant & (signed < 0)).astype(int)

    return (
        pos_per_train.sum(axis=0),
        neg_per_train.sum(axis=0),
        pos_per_train,
        neg_per_train,
    )


def _anticipation_null(inputs, *, metric, n_perm=200, rng_seed=42):
    """Per-cell null for anticipation scores under shuffled stim_cols order.

    Each iteration permutes ``stim_cols`` globally, then re-runs the same
    anticipation routine. Returns ``(pos_null, neg_null)`` of shape
    ``(n_perm, n_cells)``.
    """
    rng = np.random.default_rng(rng_seed)
    stim_cols = np.asarray(inputs["stim_cols"])
    n_cells = inputs["mat"].shape[0]
    pos_null = np.zeros((n_perm, n_cells), dtype=float)
    neg_null = np.zeros((n_perm, n_cells), dtype=float)
    for i in range(n_perm):
        shuffled = stim_cols[rng.permutation(stim_cols.size)].tolist()
        shuffled_inputs = dict(inputs)
        shuffled_inputs["stim_cols"] = shuffled
        pos, neg, _, _ = _score_anticipation(shuffled_inputs, metric=metric)
        pos_null[i] = pos
        neg_null[i] = neg
    return pos_null, neg_null


def _per_channel_responder_mask(state, exp_name, ch, cfg, threshold):
    """Boolean responder mask aligned to df.index of corrected_lum (per channel)."""
    df_indexed = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df_indexed.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
    mat = df_indexed[frame_cols].values

    F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff_mat = (mat - F0) / F0_safe

    direction = cfg.get("response_direction", "increase")
    window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
    sign = -1.0 if direction == "decrease" else 1.0
    signed_threshold = sign * float(threshold)

    stim_frames = cfg["stim_frames"].get(ch, [])
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if not stim_cols:
        return np.zeros(dff_mat.shape[0], dtype=bool)

    per_stim = np.vstack(
        [per_cell_response_delta(dff_mat, int(sc), direction, window) for sc in stim_cols]
    )
    if direction == "decrease":
        per_cell_peak = np.nanmin(per_stim, axis=0)
        return per_cell_peak <= signed_threshold
    per_cell_peak = np.nanmax(per_stim, axis=0)
    return per_cell_peak >= signed_threshold


def compute_learning_scores(
    experiments, state, *, thresholds=None, n_perm=200,
):
    """Compute habituation, sensitization, anticipation scores per DMSO expt.

    Returns a nested dict ``out[exp][metric] = {summed + per-train + responder_mask + null}``.
    Set ``n_perm <= 0`` to skip permutation-null computation.
    """
    if thresholds is None:
        thresholds = compute_responder_thresholds(experiments, state)
    out = {}
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        out[exp_name] = {}
        for metric in ("height", "width"):
            hab_chunks, sen_chunks, pos_chunks, neg_chunks = [], [], [], []
            hab_pt_chunks, sen_pt_chunks, pos_pt_chunks, neg_pt_chunks = [], [], [], []
            resp_chunks = []
            cell_id_chunks = []
            hab_null_chunks, sen_null_chunks = [], []
            pos_null_chunks, neg_null_chunks = [], []
            n_trains_seen = None
            for ch in cfg["channels"]:
                inputs = _build_learning_inputs(
                    state, exp_name, ch, cfg, metric=metric,
                )
                if inputs is None:
                    continue
                hab, hab_pt = _score_running_extremum(
                    inputs["per_stim"], mode="min",
                )
                sen, sen_pt = _score_running_extremum(
                    inputs["per_stim"], mode="max",
                )
                pos, neg, pos_pt, neg_pt = _score_anticipation(inputs, metric=metric)
                hab_chunks.append(hab)
                sen_chunks.append(sen)
                pos_chunks.append(pos)
                neg_chunks.append(neg)
                hab_pt_chunks.append(hab_pt)
                sen_pt_chunks.append(sen_pt)
                pos_pt_chunks.append(pos_pt)
                neg_pt_chunks.append(neg_pt)
                cell_id_chunks.extend((ch, cid) for cid in inputs["cell_ids"])
                if n_trains_seen is None:
                    n_trains_seen = hab_pt.shape[0]

                thr = thresholds.get((exp_name, ch))
                if thr is None:
                    resp_chunks.append(np.zeros(hab.shape[0], dtype=bool))
                else:
                    resp_chunks.append(
                        _per_channel_responder_mask(state, exp_name, ch, cfg, thr)
                    )

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
                    pos_null, neg_null = _anticipation_null(
                        inputs, metric=metric, n_perm=n_perm, rng_seed=44,
                    )
                    hab_null_chunks.append(hab_null)
                    sen_null_chunks.append(sen_null)
                    pos_null_chunks.append(pos_null)
                    neg_null_chunks.append(neg_null)

            if not hab_chunks:
                out[exp_name][metric] = None
                continue
            blob = {
                "habituation": np.concatenate(hab_chunks),
                "sensitization": np.concatenate(sen_chunks),
                "anticipation_pos": np.concatenate(pos_chunks),
                "anticipation_neg": np.concatenate(neg_chunks),
                "habituation_per_train": np.concatenate(hab_pt_chunks, axis=1),
                "sensitization_per_train": np.concatenate(sen_pt_chunks, axis=1),
                "anticipation_pos_per_train": np.concatenate(pos_pt_chunks, axis=1),
                "anticipation_neg_per_train": np.concatenate(neg_pt_chunks, axis=1),
                "responder_mask": np.concatenate(resp_chunks),
                "cell_ids": cell_id_chunks,
                "n_trains": int(n_trains_seen or 0),
            }
            if hab_null_chunks:
                blob["habituation_null"] = np.concatenate(hab_null_chunks, axis=1)
                blob["sensitization_null"] = np.concatenate(sen_null_chunks, axis=1)
                blob["anticipation_pos_null"] = np.concatenate(pos_null_chunks, axis=1)
                blob["anticipation_neg_null"] = np.concatenate(neg_null_chunks, axis=1)
                blob["habituation_pvalue"] = pvalue_one_tailed(
                    blob["habituation"], blob["habituation_null"],
                )
                blob["sensitization_pvalue"] = pvalue_one_tailed(
                    blob["sensitization"], blob["sensitization_null"],
                )
                blob["anticipation_pos_pvalue"] = pvalue_one_tailed(
                    blob["anticipation_pos"], blob["anticipation_pos_null"],
                )
                blob["anticipation_neg_pvalue"] = pvalue_one_tailed(
                    blob["anticipation_neg"], blob["anticipation_neg_null"],
                )
            out[exp_name][metric] = blob
            n = len(blob["habituation"])
            n_resp = int(blob["responder_mask"].sum())
            print(
                f"  learning scores: {exp_name} ({metric}) — {n} cells "
                f"({n_resp} responders) pooled across "
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


def _plot_score_histogram_per_train(per_train, *, title, xlabel, save_path,
                                    color=None):
    """One panel per train, side-by-side, sharing the y axis."""
    n_trains, n_cells = per_train.shape
    fig, axes = plt.subplots(
        1, n_trains,
        figsize=(4.5 * n_trains, 4.5),
        dpi=PLOT_PARAMS["dpi"],
        sharey=True,
    )
    if n_trains == 1:
        axes = np.array([axes])
    max_v = int(np.nanmax(per_train)) if per_train.size else 0
    bins = np.arange(-0.5, max_v + 1.5, 1)
    for t in range(n_trains):
        ax = axes[t]
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.hist(
            per_train[t], bins=bins,
            color=color or PLOT_PARAMS["fit_color"],
            alpha=0.85, edgecolor="#222222", linewidth=0.6,
        )
        ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
        if t == 0:
            ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_title(
            f"Train {t + 1} (n={n_cells})",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
    fig.suptitle(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"] + 1,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def _plot_score_histogram_split(scores, mask, *, title, xlabel, save_path):
    """Overlay responder vs non-responder histograms on a single axes."""
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    max_v = int(np.nanmax(scores)) if scores.size else 0
    bins = np.arange(-0.5, max_v + 1.5, 1)
    resp = scores[mask]
    non = scores[~mask]
    ax.hist(
        non, bins=bins, color="#7f7f7f", alpha=0.55,
        edgecolor="#333333", linewidth=0.6,
        label=f"Non-responder (n={non.size})",
    )
    ax.hist(
        resp, bins=bins, color="#d62728", alpha=0.65,
        edgecolor="#5a0e10", linewidth=0.6,
        label=f"Responder (n={resp.size})",
    )
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
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


def _plot_anticipation_histogram(pos, neg, *, title, save_path,
                                 pos_null=None, neg_null=None,
                                 pos_pvalues=None, neg_pvalues=None,
                                 alpha_p=0.01):
    """Overlapping pos/neg histograms with optional null overlay."""
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    max_v_candidates = [
        np.nanmax(pos) if pos.size else 0,
        np.nanmax(neg) if neg.size else 0,
    ]
    if pos_null is not None and pos_null.size:
        max_v_candidates.append(np.nanmax(pos_null))
    if neg_null is not None and neg_null.size:
        max_v_candidates.append(np.nanmax(neg_null))
    max_v = int(max(max_v_candidates))
    bins = np.arange(-0.5, max_v + 1.5, 1)

    if pos_null is not None and pos_null.size:
        n_perm = pos_null.shape[0]
        ax.hist(
            pos_null.ravel(), bins=bins,
            color="#9bd9b6", alpha=0.30,
            edgecolor="#3a7d56", linewidth=0.4,
            weights=np.full(pos_null.size, 1.0 / max(n_perm, 1)),
            label=f"pos null ({n_perm} perms)",
            zorder=1,
        )
    if neg_null is not None and neg_null.size:
        n_perm = neg_null.shape[0]
        ax.hist(
            neg_null.ravel(), bins=bins,
            color="#f5b3a8", alpha=0.30,
            edgecolor="#822c1f", linewidth=0.4,
            weights=np.full(neg_null.size, 1.0 / max(n_perm, 1)),
            label=f"neg null ({n_perm} perms)",
            zorder=1,
        )

    ax.hist(
        pos, bins=bins, color="#1a9d51", alpha=0.55,
        edgecolor="#0d4f29", linewidth=0.8,
        label=f"positive (n={pos.size})",
        zorder=2,
    )
    ax.hist(
        neg, bins=bins, color="#e74c3c", alpha=0.55,
        edgecolor="#7a1a12", linewidth=0.8,
        label=f"negative (n={neg.size})",
        zorder=2,
    )

    annotation_lines = []
    if pos_pvalues is not None and pos_pvalues.size:
        annotation_lines.append(
            f"pos p<{alpha_p}: {int(np.sum(pos_pvalues < alpha_p))} cells"
        )
    if neg_pvalues is not None and neg_pvalues.size:
        annotation_lines.append(
            f"neg p<{alpha_p}: {int(np.sum(neg_pvalues < alpha_p))} cells"
        )
    if annotation_lines:
        ax.text(
            0.98, 0.95,
            "\n".join(annotation_lines),
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"],
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
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


def _plot_anticipation_histogram_per_train(pos_pt, neg_pt, *, title, save_path):
    """Per-train side-by-side anticipation panels."""
    n_trains = pos_pt.shape[0]
    fig, axes = plt.subplots(
        1, n_trains,
        figsize=(4.5 * n_trains, 4.5),
        dpi=PLOT_PARAMS["dpi"],
        sharey=True,
    )
    if n_trains == 1:
        axes = np.array([axes])
    max_v = int(max(
        np.nanmax(pos_pt) if pos_pt.size else 0,
        np.nanmax(neg_pt) if neg_pt.size else 0,
    ))
    bins = np.arange(-0.5, max_v + 1.5, 1)
    for t in range(n_trains):
        ax = axes[t]
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.hist(
            pos_pt[t], bins=bins, color="#1a9d51", alpha=0.45,
            edgecolor="#0d4f29", linewidth=0.8,
            label=f"pos (n={pos_pt[t].size})",
        )
        ax.hist(
            neg_pt[t], bins=bins, color="#e74c3c", alpha=0.45,
            edgecolor="#7a1a12", linewidth=0.8,
            label=f"neg (n={neg_pt[t].size})",
        )
        ax.set_xlabel("Anticipation events",
                      fontsize=PLOT_PARAMS["axis_label_fontsize"])
        if t == 0:
            ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_title(
            f"Train {t + 1}",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    fig.suptitle(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"] + 1,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def _plot_anticipation_histogram_split(pos, neg, mask, *, title, save_path):
    """Overlay responder vs non-responder anticipation histograms (pos and neg)."""
    fig, axes = plt.subplots(
        1, 2,
        figsize=(PLOT_PARAMS["figsize"][0] * 1.6, PLOT_PARAMS["figsize"][1]),
        dpi=PLOT_PARAMS["dpi"],
        sharey=True,
    )
    max_v = int(max(np.nanmax(pos) if pos.size else 0,
                    np.nanmax(neg) if neg.size else 0))
    bins = np.arange(-0.5, max_v + 1.5, 1)
    for ax, scores, name, color, edge in (
        (axes[0], pos, "Positive anticipation", "#1a9d51", "#0d4f29"),
        (axes[1], neg, "Negative anticipation", "#e74c3c", "#7a1a12"),
    ):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.hist(
            scores[~mask], bins=bins, color="#7f7f7f", alpha=0.55,
            edgecolor="#333333", linewidth=0.6,
            label=f"Non-responder (n={int((~mask).sum())})",
        )
        ax.hist(
            scores[mask], bins=bins, color=color, alpha=0.65,
            edgecolor=edge, linewidth=0.6,
            label=f"Responder (n={int(mask.sum())})",
        )
        ax.set_title(
            name,
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.set_xlabel("Anticipation events",
                      fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    axes[0].set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    fig.suptitle(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"] + 1,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_learning_score_histograms(experiments, state, scores=None):
    """Emit habituation / sensitization / anticipation histograms (DMSO only).

    For each measure × metric, three figures:
        ``learning_<measure>_<metric>.png``           — summed across trains.
        ``learning_<measure>_<metric>_per_train.png`` — per-train panels.
        ``learning_<measure>_<metric>_split.png``     — responder split overlay.
    """
    if scores is None:
        scores = compute_learning_scores(experiments, state)
    for exp_name, by_metric in scores.items():
        for metric, blob in by_metric.items():
            if blob is None:
                continue
            metric_label = "height" if metric == "height" else "width"
            mask = blob["responder_mask"]

            for measure_key, score_key, pt_key, label_word in (
                ("habituation", "habituation", "habituation_per_train", "Habituation"),
                ("sensitization", "sensitization", "sensitization_per_train", "Sensitization"),
            ):
                summed = blob[score_key]
                null = blob.get(f"{measure_key}_null")
                pvals = blob.get(f"{measure_key}_pvalue")
                _plot_score_histogram(
                    summed,
                    title=(
                        f"{exp_name} — {label_word.lower()} score ({metric_label})\n"
                        f"new {'minimums' if measure_key == 'habituation' else 'maximums'} "
                        f"per cell across trains × increments"
                    ),
                    xlabel=f"{label_word} score ({metric_label})",
                    save_path=fig_path(
                        exp_name, f"learning_{measure_key}_{metric}",
                    ),
                    null_dist=null,
                    pvalues=pvals,
                )
                _plot_score_histogram_per_train(
                    blob[pt_key],
                    title=(
                        f"{exp_name} — {label_word.lower()} per train ({metric_label})"
                    ),
                    xlabel=f"{label_word} score / train",
                    save_path=fig_path(
                        exp_name, f"learning_{measure_key}_{metric}_per_train",
                    ),
                )
                _plot_score_histogram_split(
                    summed, mask,
                    title=(
                        f"{exp_name} — {label_word.lower()} score ({metric_label})\n"
                        f"responders vs non-responders"
                    ),
                    xlabel=f"{label_word} score ({metric_label})",
                    save_path=fig_path(
                        exp_name, f"learning_{measure_key}_{metric}_split",
                    ),
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
                pos_null=blob.get("anticipation_pos_null"),
                neg_null=blob.get("anticipation_neg_null"),
                pos_pvalues=blob.get("anticipation_pos_pvalue"),
                neg_pvalues=blob.get("anticipation_neg_pvalue"),
            )
            _plot_anticipation_histogram_per_train(
                blob["anticipation_pos_per_train"],
                blob["anticipation_neg_per_train"],
                title=(
                    f"{exp_name} — anticipation per train ({metric_label})"
                ),
                save_path=fig_path(
                    exp_name, f"learning_anticipation_{metric}_per_train",
                ),
            )
            _plot_anticipation_histogram_split(
                blob["anticipation_pos"],
                blob["anticipation_neg"],
                mask,
                title=(
                    f"{exp_name} — anticipation score ({metric_label})\n"
                    f"responders vs non-responders"
                ),
                save_path=fig_path(
                    exp_name, f"learning_anticipation_{metric}_split",
                ),
            )


def plot_learning_score_histograms_per_cluster(experiments, state, scores=None):
    """For each cluster (per-channel cache): summed habituation/sensitization/anticipation."""
    if scores is None:
        scores = compute_learning_scores(experiments, state)
    for exp_name, by_metric in scores.items():
        cfg = experiments[exp_name]
        for ch in cfg["channels"]:
            blob = load_cluster_labels(exp_name, ch)
            if blob is None:
                continue
            best_k = int(blob["best_k"])
            for metric, mblob in by_metric.items():
                if mblob is None:
                    continue
                metric_label = "height" if metric == "height" else "width"
                cluster_labels = align_labels_to_cells(
                    blob, mblob["cell_ids"],
                )
                for cluster_id in range(best_k):
                    cell_mask = cluster_labels == cluster_id
                    if int(cell_mask.sum()) < 3:
                        continue
                    hab_sub = mblob["habituation"][cell_mask]
                    sen_sub = mblob["sensitization"][cell_mask]
                    pos_sub = mblob["anticipation_pos"][cell_mask]
                    neg_sub = mblob["anticipation_neg"][cell_mask]
                    suffix = f"_c{cluster_id}_{ch}".replace(" ", "_")
                    _plot_score_histogram(
                        hab_sub,
                        title=(
                            f"{exp_name} / {ch} cluster {cluster_id} — "
                            f"habituation ({metric_label}; n={int(cell_mask.sum())})"
                        ),
                        xlabel=f"Habituation score ({metric_label})",
                        save_path=fig_path(
                            exp_name, f"learning_habituation_{metric}{suffix}",
                        ),
                    )
                    _plot_score_histogram(
                        sen_sub,
                        title=(
                            f"{exp_name} / {ch} cluster {cluster_id} — "
                            f"sensitization ({metric_label}; n={int(cell_mask.sum())})"
                        ),
                        xlabel=f"Sensitization score ({metric_label})",
                        save_path=fig_path(
                            exp_name, f"learning_sensitization_{metric}{suffix}",
                        ),
                    )
                    _plot_anticipation_histogram(
                        pos_sub, neg_sub,
                        title=(
                            f"{exp_name} / {ch} cluster {cluster_id} — "
                            f"anticipation ({metric_label}; n={int(cell_mask.sum())})"
                        ),
                        save_path=fig_path(
                            exp_name, f"learning_anticipation_{metric}{suffix}",
                        ),
                    )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    scores = compute_learning_scores(experiments, state)
    plot_learning_score_histograms(experiments, state, scores=scores)
    plot_learning_score_histograms_per_cluster(experiments, state, scores=scores)


if __name__ == "__main__":
    main()
