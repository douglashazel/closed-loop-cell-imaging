#!/usr/bin/env python3
"""Pairwise correlation vs spatial distance.

Per experiment:
    * corr_vs_dist.png            — per-channel scatter panels
    * corr_vs_dist_combined.png   — pooled across channels
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
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


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


def _scatter_corr_vs_dist(ax, dists, corrs, color, title):
    """Helper: scatter pairwise (distance, correlation) with a fitted line + ±3 SEM band."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter(dists, corrs, color=color, alpha=0.35, s=8, edgecolors="none", zorder=1)

    valid = ~np.isnan(dists) & ~np.isnan(corrs)
    if np.sum(valid) > 2:
        xv, yv = dists[valid], corrs[valid]
        res = linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 100)
        y_line = res.slope * x_line + res.intercept

        dof = len(xv) - 2
        rss = np.sum((yv - (res.slope * xv + res.intercept)) ** 2)
        rse = np.sqrt(rss / dof) if dof > 0 else np.nan
        mean_x = np.mean(xv)
        ssx = np.sum((xv - mean_x) ** 2)
        y_err = (
            rse * np.sqrt(1 / len(xv) + (x_line - mean_x) ** 2 / ssx)
            if ssx > 0 else np.zeros_like(x_line)
        )

        ax.plot(
            x_line, y_line,
            color=PLOT_PARAMS["corr_fit_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=(
                f"Linear fit\n"
                f"Pearson r = {res.rvalue:.3f}\n"
                f"R² = {res.rvalue ** 2:.3f}\n"
                f"p (slope=0) = {res.pvalue:.2e}"
            ),
            zorder=3,
        )
        ax.fill_between(
            x_line, y_line - 3 * y_err, y_line + 3 * y_err,
            color=PLOT_PARAMS["corr_band_color"], alpha=0.30, zorder=2,
            label="±3 SEM",
        )

    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")


def _plot_corr_vs_dist_combined(exp_name, per_channel_pairs):
    """Single-axes scatter pooling pairwise (distance, correlation) across channels."""
    if not per_channel_pairs:
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=PLOT_PARAMS["dpi"])
    ax.spines[["top", "right"]].set_visible(False)

    all_dist, all_corr = [], []
    for col, (ch, pw_dist, pw_corr, n_cells) in enumerate(per_channel_pairs):
        c = PLOT_PARAMS["corr_scatter_colors"][col % len(PLOT_PARAMS["corr_scatter_colors"])]
        ax.scatter(
            pw_dist, pw_corr,
            color=c, alpha=0.30, s=8, edgecolors="none", zorder=1,
            label=f"{ch} ({n_cells} cells)",
        )
        all_dist.append(np.asarray(pw_dist))
        all_corr.append(np.asarray(pw_corr))

    dists = np.concatenate(all_dist)
    corrs = np.concatenate(all_corr)
    valid = ~np.isnan(dists) & ~np.isnan(corrs)
    if np.sum(valid) > 2:
        xv, yv = dists[valid], corrs[valid]
        res = linregress(xv, yv)
        x_line = np.linspace(xv.min(), xv.max(), 200)
        y_line = res.slope * x_line + res.intercept

        dof = len(xv) - 2
        rss = np.sum((yv - (res.slope * xv + res.intercept)) ** 2)
        rse = np.sqrt(rss / dof) if dof > 0 else np.nan
        mean_x = np.mean(xv)
        ssx = np.sum((xv - mean_x) ** 2)
        y_err = (
            rse * np.sqrt(1 / len(xv) + (x_line - mean_x) ** 2 / ssx)
            if ssx > 0 else np.zeros_like(x_line)
        )

        ax.plot(
            x_line, y_line,
            color=PLOT_PARAMS["corr_fit_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=(
                f"Pooled linear fit\n"
                f"Pearson r = {res.rvalue:.3f}\n"
                f"R² = {res.rvalue ** 2:.3f}\n"
                f"p (slope=0) = {res.pvalue:.2e}"
            ),
            zorder=3,
        )
        ax.fill_between(
            x_line, y_line - 3 * y_err, y_line + 3 * y_err,
            color=PLOT_PARAMS["corr_band_color"], alpha=0.30, zorder=2,
            label="±3 SEM",
        )

    ax.set_xlabel(
        "Pairwise distance (px)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_ylabel(
        "Pearson r (full corrected time series)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_title(
        f"{exp_name} — pairwise correlation vs distance (all channels combined)",
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
    """One figure per experiment: pairwise Pearson r vs spatial distance."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        fig, axes = plt.subplots(
            1, len(channels),
            figsize=(6 * len(channels), 6),
            dpi=PLOT_PARAMS["dpi"], sharey=True,
        )
        if len(channels) == 1:
            axes = np.array([axes])

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
                axes[col].set_title(
                    f"{ch}: insufficient data",
                    fontsize=PLOT_PARAMS["title_fontsize"],
                )
                continue

            corr_mat = pd.DataFrame(mat_k).T.corr(method="pearson").values
            dist_mat = squareform(pdist(pos_xy, metric="euclidean"))
            iu = np.triu_indices(corr_mat.shape[0], k=1)
            pw_corr = corr_mat[iu]
            pw_dist = dist_mat[iu]

            per_channel_pairs.append((ch, pw_dist, pw_corr, len(keep_rows)))

            _scatter_corr_vs_dist(
                axes[col], pw_dist, pw_corr,
                color=PLOT_PARAMS["corr_scatter_colors"][col % len(PLOT_PARAMS["corr_scatter_colors"])],
                title=f"{ch}  ({len(keep_rows)} cells)",
            )
            axes[col].set_xlabel(
                "Pairwise distance (px)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            if col == 0:
                axes[col].set_ylabel(
                    "Pearson r (full corrected time series)",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )

        fig.suptitle(
            f"{exp_name} — pairwise correlation vs pairwise distance",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "corr_vs_dist"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)

        _plot_corr_vs_dist_combined(exp_name, per_channel_pairs)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_correlation_vs_distance(experiments, state)


if __name__ == "__main__":
    main()
