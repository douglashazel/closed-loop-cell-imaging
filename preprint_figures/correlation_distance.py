#!/usr/bin/env python3
"""Pairwise correlation vs spatial distance.

Per experiment:
    * corr_vs_dist.png            — per-channel scatter panels (2 rows: Pearson, Spearman)
    * corr_vs_dist_combined.png   — pooled across channels (Pearson + Spearman, stacked)

Within each panel, pairs are coloured by responder-pair status (RR / NN / RN)
when responder thresholds are available, with a separate regression line per
subset. Falls back to a single-cloud scatter when no responder threshold
applies for the channel.
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
from common.config import PEAK_OFFSET
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_thresholds
from common.stim_helpers import compute_f0_baseline, per_cell_response_delta

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


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


def _plot_corr_vs_dist_combined(exp_name, per_channel_pairs):
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
            "Pairwise distance (px)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_ylabel(
            f"{METHOD_LABEL[method]} (full corrected time series)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_title(
            f"{exp_name} — pairwise {METHOD_LABEL[method]} vs distance "
            f"(all channels combined)",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

    plt.tight_layout()
    fig.savefig(
        fig_path(exp_name, "corr_vs_dist_combined"),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)


def plot_correlation_vs_distance(experiments, state):
    """Per experiment: Pearson + Spearman correlation vs pairwise distance.

    Two-row figure with Pearson on row 0 and Spearman on row 1, one column
    per channel. Pairs are colored by responder-pair class when responder
    thresholds are available.
    """
    thresholds = compute_responder_thresholds(experiments, state)

    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        fig, axes = plt.subplots(
            2, len(channels),
            figsize=(6 * len(channels), 11),
            dpi=PLOT_PARAMS["dpi"], sharey="row",
        )
        if len(channels) == 1:
            axes = axes.reshape(2, 1)

        per_channel_pairs = []

        for col, ch in enumerate(channels):
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
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
            mat_k = mat[keep_rows]
            pos_xy = np.array(pos_xy, dtype=float)

            if len(pos_xy) < 2:
                for row in range(2):
                    axes[row, col].set_title(
                        f"{ch}: insufficient data",
                        fontsize=PLOT_PARAMS["title_fontsize"],
                    )
                continue

            pearson_mat = pd.DataFrame(mat_k).T.corr(method="pearson").values
            spearman_mat = pd.DataFrame(mat_k).T.corr(method="spearman").values
            dist_mat = squareform(pdist(pos_xy, metric="euclidean"))
            iu = np.triu_indices(pearson_mat.shape[0], k=1)
            pw_dist = dist_mat[iu]
            pw_corr_by_method = {
                "pearson": pearson_mat[iu],
                "spearman": spearman_mat[iu],
            }

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

            per_channel_pairs.append({
                "ch": ch,
                "n_cells": len(keep_rows),
                "pw_dist": pw_dist,
                "pw_corr_by_method": pw_corr_by_method,
                "pair_classes": pair_classes,
            })

            for row, method in enumerate(("pearson", "spearman")):
                _scatter_corr_vs_dist(
                    axes[row, col],
                    pw_dist, pw_corr_by_method[method],
                    pair_classes=pair_classes,
                    corr_method=method,
                    title=f"{ch}  ({len(keep_rows)} cells, {METHOD_LABEL[method]})",
                )
                axes[row, col].set_xlabel(
                    "Pairwise distance (px)",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
                if col == 0:
                    axes[row, col].set_ylabel(
                        f"{METHOD_LABEL[method]} (full corrected time series)",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )

        fig.suptitle(
            f"{exp_name} — pairwise correlation vs pairwise distance "
            f"(Pearson top, Spearman bottom)",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "corr_vs_dist"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)

        _plot_corr_vs_dist_combined(exp_name, per_channel_pairs)


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

            for cid in range(best_k):
                in_cluster = cluster_labels_k == cid
                if int(in_cluster.sum()) < 5:
                    continue
                sub_mat = mat_k[in_cluster]
                sub_pos = pos_xy[in_cluster]
                pearson_mat = pd.DataFrame(sub_mat).T.corr(method="pearson").values
                spearman_mat = pd.DataFrame(sub_mat).T.corr(method="spearman").values
                dist_mat = squareform(pdist(sub_pos, metric="euclidean"))
                iu = np.triu_indices(pearson_mat.shape[0], k=1)
                pw_dist = dist_mat[iu]
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
                    "Pairwise distance (px)",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
                axes[0].set_ylabel(
                    "Pearson r",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
                _scatter_corr_vs_dist(
                    axes[1], pw_dist, pw_corr_spearman,
                    pair_classes=None, corr_method="spearman",
                    title=f"Spearman  (n={int(in_cluster.sum())} cells in cluster)",
                )
                axes[1].set_xlabel(
                    "Pairwise distance (px)",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
                axes[1].set_ylabel(
                    "Spearman ρ",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
                fig.suptitle(
                    f"{exp_name} / {ch} — cluster {cid} — "
                    f"pairwise correlation vs distance",
                    fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                    fontweight="bold", y=1.01,
                )
                plt.tight_layout()
                fig.savefig(
                    fig_path(exp_name, f"{ch}_corr_vs_dist_c{cid}"),
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
