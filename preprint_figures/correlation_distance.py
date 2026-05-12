#!/usr/bin/env python3
"""Pairwise correlation vs spatial distance.

Per experiment, three time-window variants of each plot are produced — full
time series, during-stim/train windows, and between-stim/train windows. The
during/between variants are skipped silently when a channel has no stim
onsets.

    * corr_vs_dist[{_during,_between}].png         — per-channel scatter panels
    * corr_vs_dist_combined[{_during,_between}].png — pooled across channels
    * {ch}_corr_vs_dist_c{cid}[{_during,_between}].png — per-cluster panels

Within each panel, pairs are coloured by responder-pair status (RR / NN / RN)
when responder thresholds are available, with a separate regression line per
subset. Falls back to a single-cloud scatter when no responder threshold
applies for the channel. Distances are reported in μm using the imaging
calibration ``PIXELS_PER_UM``.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.cluster_labels import align_labels_to_cells, load_cluster_labels
from common.config import LEARNING_STIMS_PER_TRAIN, PEAK_OFFSET
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_thresholds
from common.stim_helpers import compute_f0_baseline, per_cell_response_delta
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


PIXELS_PER_UM = 180.1  # imaging calibration: 0.00555 μm/pixel
RESPONSE_PAD_TRAIN_MIN = 10.0
RESPONSE_PAD_STIM_MIN = 1.5
MIN_FRAMES_FOR_CORR = 5
PAIR_CLASS_COLORS = {
    "RR": "#d62728",   # red — both responders
    "NN": "#7f7f7f",   # gray — both non-responders
    "RN": "#2ca02c",   # green — mixed
}
PAIR_CLASS_LABEL = {
    "RR": "Responder × Responder",
    "NN": "Non × Non",
    "RN": "Responder × Non",
}
METHOD_LABEL = {"pearson": "Pearson r", "spearman": "Spearman ρ"}


def mean_cell_positions(traj, n_frames):
    """Return ``{cell_id_str: (mean_x, mean_y)}`` averaged over valid frames."""
    positions = {}
    for cid, coords in traj.items():
        xs, ys = [], []
        for i in range(n_frames):
            x = coords.get(f"x{i}")
            y = coords.get(f"y{i}")
            if x is None or y is None:
                continue
            try:
                xs.append(float(x))
                ys.append(float(y))
            except (TypeError, ValueError):
                continue
        if xs:
            positions[cid] = (float(np.mean(xs)), float(np.mean(ys)))
    return positions


def _per_cell_responder_mask(state, exp_name, ch, cfg, threshold):
    """Boolean responder mask aligned to ``df.index`` ordering of corrected_lum.

    Recipe matches response_violins.py — peak |Δ dF/F0| over the union of
    response windows (one per stim) compared against the Bonferroni threshold.
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
        return np.zeros(dff_mat.shape[0], dtype=bool), list(df_indexed.index)

    per_stim = np.vstack(
        [per_cell_response_delta(dff_mat, int(sc), direction, window) for sc in stim_cols]
    )
    if direction == "decrease":
        per_cell_peak = np.nanmin(per_stim, axis=0)
        mask = per_cell_peak <= signed_threshold
    else:
        per_cell_peak = np.nanmax(per_stim, axis=0)
        mask = per_cell_peak >= signed_threshold
    return mask, list(df_indexed.index)


def _classify_pair_classes(responder_mask):
    """Return a dict of pair-class → 1-D boolean mask aligned to triu_indices(k=1).

    For ``n`` cells, the upper-triangle iu has ``n*(n-1)/2`` entries. Each
    pair (i, j) is classified as RR (both True), NN (both False), or RN (mix).
    """
    n = len(responder_mask)
    if n < 2:
        return {"RR": np.array([], dtype=bool), "NN": np.array([], dtype=bool), "RN": np.array([], dtype=bool)}
    iu = np.triu_indices(n, k=1)
    a = responder_mask[iu[0]]
    b = responder_mask[iu[1]]
    return {
        "RR": a & b,
        "NN": (~a) & (~b),
        "RN": a ^ b,
    }


def _time_window_masks(cfg, ch, frame_nums, state, exp_name):
    """Return ``[(key, label, mask_or_None), ...]`` describing time-axis windows.

    Always emits the ``"full"`` window (mask is ``None`` → use every column).
    When the channel has stim onsets, additionally emits ``"during"`` and
    ``"between"`` windows. For experiments whose stim count is a positive
    multiple of ``LEARNING_STIMS_PER_TRAIN`` the during-mask spans entire
    trains (first-stim onset through ``RESPONSE_PAD_TRAIN_MIN`` minutes after
    the last-stim onset). Otherwise each stim contributes its own window of
    ``stim_duration_minutes + RESPONSE_PAD_STIM_MIN`` minutes.
    """
    full = ("full", "full time series", None)

    stim_frames = cfg.get("stim_frames", {}).get(ch, []) or []
    if not stim_frames:
        return [full]

    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if not stim_cols:
        return [full]

    n_cols = len(frame_nums)
    col_minutes = frames_to_min(state, exp_name, ch, np.array(frame_nums, dtype=float))
    during = np.zeros(n_cols, dtype=bool)

    is_trains = (
        len(stim_cols) >= LEARNING_STIMS_PER_TRAIN
        and len(stim_cols) % LEARNING_STIMS_PER_TRAIN == 0
    )

    if is_trains:
        n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
        for t in range(n_trains):
            first_sc = stim_cols[t * LEARNING_STIMS_PER_TRAIN]
            last_sc = stim_cols[(t + 1) * LEARNING_STIMS_PER_TRAIN - 1]
            end_min = float(col_minutes[last_sc]) + RESPONSE_PAD_TRAIN_MIN
            end_col = int(np.argmin(np.abs(col_minutes - end_min)))
            during[first_sc:end_col + 1] = True
        during_label = "during trains"
        between_label = "between trains"
    else:
        stim_duration_min = float(cfg.get("stim_duration_minutes", 0.0) or 0.0)
        pad = stim_duration_min + RESPONSE_PAD_STIM_MIN
        for sc in stim_cols:
            end_min = float(col_minutes[sc]) + pad
            end_col = int(np.argmin(np.abs(col_minutes - end_min)))
            during[sc:end_col + 1] = True
        during_label = "during stim"
        between_label = "between stim"

    return [
        full,
        ("during", during_label, during),
        ("between", between_label, ~during),
    ]


def _fit_and_plot_subset(ax, dists, corrs, color, label_prefix):
    """Fit a line on (dists, corrs); plot scatter + line + ±3 SEM band.

    Returns True if a line was drawn, False if too few valid points.
    """
    valid = ~np.isnan(dists) & ~np.isnan(corrs)
    n = int(valid.sum())
    if n < 3:
        if n > 0:
            ax.scatter(
                dists[valid], corrs[valid],
                color=color, alpha=0.35, s=8, edgecolors="none",
                label=f"{label_prefix} (n={n})",
                zorder=1,
            )
        return False
    xv, yv = dists[valid], corrs[valid]
    res = linregress(xv, yv)
    x_line = np.linspace(xv.min(), xv.max(), 100)
    y_line = res.slope * x_line + res.intercept

    ax.scatter(
        xv, yv,
        color=color, alpha=0.30, s=8, edgecolors="none",
        zorder=1,
    )

    dof = len(xv) - 2
    rss = np.sum((yv - (res.slope * xv + res.intercept)) ** 2)
    rse = np.sqrt(rss / dof) if dof > 0 else np.nan
    mean_x = np.mean(xv)
    ssx = np.sum((xv - mean_x) ** 2)
    if ssx > 0 and not np.isnan(rse):
        y_err = rse * np.sqrt(1 / len(xv) + (x_line - mean_x) ** 2 / ssx)
        ax.fill_between(
            x_line, y_line - 3 * y_err, y_line + 3 * y_err,
            color=color, alpha=0.10, zorder=1.5,
        )

    ax.plot(
        x_line, y_line,
        color=color,
        linewidth=PLOT_PARAMS["mean_lw"],
        label=(
            f"{label_prefix} (n={n})\n"
            f"slope={res.slope:.2e}  r={res.rvalue:.3f}  "
            f"p={res.pvalue:.2e}"
        ),
        zorder=3,
    )
    return True


def _scatter_corr_vs_dist(
    ax, pw_dist, pw_corr, *, pair_classes=None,
    corr_method="pearson", title="",
):
    """Scatter pairwise (distance, correlation), optionally split by pair class.

    When ``pair_classes`` is None or all-empty, falls back to a single fit.
    """
    ax.spines[["top", "right"]].set_visible(False)
    drew_any = False
    if pair_classes and any(m.any() for m in pair_classes.values()):
        for tag in ("NN", "RN", "RR"):
            mask = pair_classes.get(tag)
            if mask is None or not mask.any():
                continue
            drew_any |= _fit_and_plot_subset(
                ax,
                pw_dist[mask], pw_corr[mask],
                color=PAIR_CLASS_COLORS[tag],
                label_prefix=PAIR_CLASS_LABEL[tag],
            )
    else:
        drew_any = _fit_and_plot_subset(
            ax,
            np.asarray(pw_dist, dtype=float),
            np.asarray(pw_corr, dtype=float),
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All pairs",
        )

    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    if drew_any:
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")


def _plot_corr_vs_dist_combined(
    exp_name, per_channel_pairs, *, window_label="full corrected time series",
    suffix="",
):
    """Two stacked axes (Pearson on top, Spearman below) pooling across channels."""
    if not per_channel_pairs:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 11), dpi=PLOT_PARAMS["dpi"])

    for row, method in enumerate(("pearson", "spearman")):
        ax = axes[row]
        ax.spines[["top", "right"]].set_visible(False)

        all_dist = []
        all_corr = []
        all_classes = {"RR": [], "NN": [], "RN": []}
        for col, entry in enumerate(per_channel_pairs):
            ch = entry["ch"]
            n_cells = entry["n_cells"]
            pw_dist = entry["pw_dist"]
            pw_corr = entry["pw_corr_by_method"][method]
            pair_classes = entry.get("pair_classes")

            color = PLOT_PARAMS["corr_scatter_colors"][
                col % len(PLOT_PARAMS["corr_scatter_colors"])
            ]
            ax.scatter(
                pw_dist, pw_corr,
                color=color, alpha=0.18, s=6, edgecolors="none", zorder=1,
                label=f"{ch} ({n_cells} cells)",
            )
            all_dist.append(np.asarray(pw_dist))
            all_corr.append(np.asarray(pw_corr))
            if pair_classes is not None:
                for tag in all_classes:
                    all_classes[tag].append(pair_classes[tag])

        dists = np.concatenate(all_dist)
        corrs = np.concatenate(all_corr)
        any_classes = any(len(v) > 0 for v in all_classes.values())
        merged_classes = (
            {tag: np.concatenate(v) for tag, v in all_classes.items()}
            if any_classes else None
        )

        if merged_classes is not None and any(m.any() for m in merged_classes.values()):
            for tag in ("NN", "RN", "RR"):
                mask = merged_classes.get(tag)
                if mask is None or not mask.any():
                    continue
                _fit_and_plot_subset(
                    ax,
                    dists[mask], corrs[mask],
                    color=PAIR_CLASS_COLORS[tag],
                    label_prefix=f"{PAIR_CLASS_LABEL[tag]} (pooled)",
                )
        else:
            _fit_and_plot_subset(
                ax,
                dists, corrs,
                color=PLOT_PARAMS["corr_fit_color"],
                label_prefix="Pooled",
            )

        ax.set_xlabel(
            "Pairwise distance (μm)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_ylabel(
            f"{METHOD_LABEL[method]} ({window_label})",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_title(
            f"{exp_name} — pairwise {METHOD_LABEL[method]} vs distance "
            f"({window_label}, all channels combined)",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

    plt.tight_layout()
    fig.savefig(
        fig_path(exp_name, f"corr_vs_dist_combined{suffix}"),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)


def plot_correlation_vs_distance(experiments, state):
    """Per experiment: Pearson + Spearman correlation vs pairwise distance.

    Two-row figure with Pearson on row 0 and Spearman on row 1, one column
    per channel. Pairs are colored by responder-pair class when responder
    thresholds are available. Three figures are produced per experiment:
    full time series, during-stim/train windows, and between-stim/train
    windows. The latter two are skipped silently when no channel has stim
    onsets.
    """
    thresholds = compute_responder_thresholds(experiments, state)

    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        # Build per-channel context once (window-independent).
        per_channel_ctx = {}
        for ch in channels:
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df[frame_cols].values
            cell_ids_int = list(df.index)

            positions = mean_cell_positions(
                state["traj_by_channel"][exp_name][ch],
                state["frame_counts"][exp_name][ch],
            )
            keep_rows, pos_xy = [], []
            for r, cid_int in enumerate(cell_ids_int):
                for key in (str(cid_int), cid_int):
                    if key in positions:
                        keep_rows.append(r)
                        pos_xy.append(positions[key])
                        break
            pos_xy = np.array(pos_xy, dtype=float)

            if len(pos_xy) < 2:
                per_channel_ctx[ch] = None
                continue

            mat_k = mat[keep_rows]
            dist_mat = squareform(pdist(pos_xy, metric="euclidean"))
            iu = np.triu_indices(len(pos_xy), k=1)
            pw_dist = dist_mat[iu] / PIXELS_PER_UM

            pair_classes = None
            thr = thresholds.get((exp_name, ch))
            if thr is not None:
                full_mask, full_ids = _per_cell_responder_mask(
                    state, exp_name, ch, cfg, thr,
                )
                id_to_mask = dict(zip(full_ids, full_mask))
                row_mask = np.array(
                    [bool(id_to_mask.get(cell_ids_int[r], False)) for r in keep_rows],
                    dtype=bool,
                )
                pair_classes = _classify_pair_classes(row_mask)

            per_channel_ctx[ch] = {
                "mat_k": mat_k,
                "pw_dist": pw_dist,
                "iu": iu,
                "n_cells": len(keep_rows),
                "pair_classes": pair_classes,
                "windows": {
                    w_key: (w_label, w_mask)
                    for w_key, w_label, w_mask in _time_window_masks(
                        cfg, ch, frame_nums, state, exp_name,
                    )
                },
            }

        # Collect every window key that appears in any channel.
        ordered_keys = ["full", "during", "between"]
        active_keys = [
            k for k in ordered_keys
            if any(
                ctx is not None and k in ctx["windows"]
                for ctx in per_channel_ctx.values()
            )
        ]

        for w_key in active_keys:
            suffix = "" if w_key == "full" else f"_{w_key}"
            fig, axes = plt.subplots(
                2, len(channels),
                figsize=(6 * len(channels), 11),
                dpi=PLOT_PARAMS["dpi"], sharey="row",
            )
            if len(channels) == 1:
                axes = axes.reshape(2, 1)

            per_channel_pairs = []
            window_label_for_fig = None

            for col, ch in enumerate(channels):
                ctx = per_channel_ctx.get(ch)
                if ctx is None:
                    for row in range(2):
                        axes[row, col].set_title(
                            f"{ch}: insufficient data",
                            fontsize=PLOT_PARAMS["title_fontsize"],
                        )
                    continue

                window_entry = ctx["windows"].get(w_key)
                if window_entry is None:
                    for row in range(2):
                        axes[row, col].set_title(
                            f"{ch}: no stim windows",
                            fontsize=PLOT_PARAMS["title_fontsize"],
                        )
                    continue

                window_label, time_mask = window_entry
                if window_label_for_fig is None:
                    window_label_for_fig = window_label

                mat_k = ctx["mat_k"]
                mat_k_w = mat_k if time_mask is None else mat_k[:, time_mask]
                if mat_k_w.shape[1] < MIN_FRAMES_FOR_CORR:
                    for row in range(2):
                        axes[row, col].set_title(
                            f"{ch}: insufficient {window_label} samples",
                            fontsize=PLOT_PARAMS["title_fontsize"],
                        )
                    continue

                pearson_mat = pd.DataFrame(mat_k_w).T.corr(method="pearson").values
                spearman_mat = pd.DataFrame(mat_k_w).T.corr(method="spearman").values
                iu = ctx["iu"]
                pw_corr_by_method = {
                    "pearson": pearson_mat[iu],
                    "spearman": spearman_mat[iu],
                }

                per_channel_pairs.append({
                    "ch": ch,
                    "n_cells": ctx["n_cells"],
                    "pw_dist": ctx["pw_dist"],
                    "pw_corr_by_method": pw_corr_by_method,
                    "pair_classes": ctx["pair_classes"],
                })

                for row, method in enumerate(("pearson", "spearman")):
                    _scatter_corr_vs_dist(
                        axes[row, col],
                        ctx["pw_dist"], pw_corr_by_method[method],
                        pair_classes=ctx["pair_classes"],
                        corr_method=method,
                        title=f"{ch}  ({ctx['n_cells']} cells, {METHOD_LABEL[method]})",
                    )
                    axes[row, col].set_xlabel(
                        "Pairwise distance (μm)",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )
                    if col == 0:
                        axes[row, col].set_ylabel(
                            f"{METHOD_LABEL[method]} ({window_label})",
                            fontsize=PLOT_PARAMS["axis_label_fontsize"],
                        )

            if window_label_for_fig is None:
                window_label_for_fig = w_key
            fig.suptitle(
                f"{exp_name} — pairwise correlation vs pairwise distance "
                f"({window_label_for_fig}, Pearson top, Spearman bottom)",
                fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                fontweight="bold", y=1.01,
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"corr_vs_dist{suffix}"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)

            _plot_corr_vs_dist_combined(
                exp_name, per_channel_pairs,
                window_label=window_label_for_fig, suffix=suffix,
            )


def plot_correlation_vs_distance_per_cluster(experiments, state):
    """Per (exp, ch, cluster): pairwise Pearson + Spearman vs distance.

    Restricts each panel to within-cluster cell pairs. Skips clusters with
    fewer than 5 cells. Silently no-ops when no cluster cache exists.
    """
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            blob = load_cluster_labels(exp_name, ch)
            if blob is None:
                continue
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df[frame_cols].values
            cell_ids_int = list(df.index)
            target_ids = [(ch, cid) for cid in cell_ids_int]
            cluster_labels = align_labels_to_cells(blob, target_ids)

            positions = mean_cell_positions(
                state["traj_by_channel"][exp_name][ch],
                state["frame_counts"][exp_name][ch],
            )
            keep_rows, pos_xy = [], []
            for r, cid_int in enumerate(cell_ids_int):
                for key in (str(cid_int), cid_int):
                    if key in positions:
                        keep_rows.append(r)
                        pos_xy.append(positions[key])
                        break
            if len(keep_rows) < 2:
                continue
            mat_k = mat[keep_rows]
            pos_xy = np.array(pos_xy, dtype=float)
            cluster_labels_k = cluster_labels[keep_rows]
            best_k = int(blob["best_k"])

            windows = _time_window_masks(cfg, ch, frame_nums, state, exp_name)

            for cid in range(best_k):
                in_cluster = cluster_labels_k == cid
                if int(in_cluster.sum()) < 5:
                    continue
                sub_mat = mat_k[in_cluster]
                sub_pos = pos_xy[in_cluster]
                dist_mat = squareform(pdist(sub_pos, metric="euclidean"))
                iu = np.triu_indices(sub_mat.shape[0], k=1)
                pw_dist = dist_mat[iu] / PIXELS_PER_UM

                for w_key, w_label, w_mask in windows:
                    suffix = "" if w_key == "full" else f"_{w_key}"
                    sub_mat_w = sub_mat if w_mask is None else sub_mat[:, w_mask]
                    if sub_mat_w.shape[1] < MIN_FRAMES_FOR_CORR:
                        continue

                    pearson_mat = pd.DataFrame(sub_mat_w).T.corr(method="pearson").values
                    spearman_mat = pd.DataFrame(sub_mat_w).T.corr(method="spearman").values
                    pw_corr_pearson = pearson_mat[iu]
                    pw_corr_spearman = spearman_mat[iu]

                    fig, axes = plt.subplots(
                        2, 1, figsize=(7, 11), dpi=PLOT_PARAMS["dpi"],
                    )
                    _scatter_corr_vs_dist(
                        axes[0], pw_dist, pw_corr_pearson,
                        pair_classes=None, corr_method="pearson",
                        title=f"Pearson  (n={int(in_cluster.sum())} cells in cluster)",
                    )
                    axes[0].set_xlabel(
                        "Pairwise distance (μm)",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )
                    axes[0].set_ylabel(
                        f"Pearson r ({w_label})",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )
                    _scatter_corr_vs_dist(
                        axes[1], pw_dist, pw_corr_spearman,
                        pair_classes=None, corr_method="spearman",
                        title=f"Spearman  (n={int(in_cluster.sum())} cells in cluster)",
                    )
                    axes[1].set_xlabel(
                        "Pairwise distance (μm)",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )
                    axes[1].set_ylabel(
                        f"Spearman ρ ({w_label})",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )
                    fig.suptitle(
                        f"{exp_name} / {ch} — cluster {cid} — "
                        f"pairwise correlation vs distance ({w_label})",
                        fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                        fontweight="bold", y=1.01,
                    )
                    plt.tight_layout()
                    fig.savefig(
                        fig_path(exp_name, f"{ch}_corr_vs_dist_c{cid}{suffix}"),
                        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
                    )
                    plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_correlation_vs_distance(experiments, state)
    plot_correlation_vs_distance_per_cluster(experiments, state)


if __name__ == "__main__":
    main()
