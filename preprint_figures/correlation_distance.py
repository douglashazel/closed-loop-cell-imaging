#!/usr/bin/env python3
"""Pairwise correlation vs spatial distance (full time series).

Per experiment:
    * corr_vs_dist.png           — per-channel scatter panels
    * corr_vs_dist_combined.png  — pooled across channels

Both use the full corrected time series. Within each panel, pairs are coloured
by responder-pair status (RR / NN / RN) when responder thresholds are
available, with a separate descriptive regression line per subset. Falls back
to a single-cloud scatter when no responder threshold applies for the channel.
Significance is a Mantel permutation test (per channel) — the regression
p-value is invalid here because the ~N²/2 cell pairs are not independent.
Distances are reported in μm using the imaging calibration ``PIXELS_PER_UM``.
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
from common.io_paths import fig_path, save_fig
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_masks
from common.stats import inferential_caveat, mantel_test, one_sample_t_dz

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


PIXELS_PER_UM = 180.1  # imaging calibration: 0.00555 μm/pixel
MIN_FRAMES_FOR_CORR = 5
PAIR_CLASS_COLORS = {
    "RR": PLOT_PARAMS["rr_color"],   # blue — both responders
    "NN": "#7f7f7f",                 # gray — both non-responders
}
PAIR_CLASS_LABEL = {
    "RR": "Responder × Responder",
    "NN": "Non-responder pairs",
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


def _fit_and_plot_subset(ax, dists, corrs, color, label_prefix, *, mantel=None):
    """Fit a line on (dists, corrs); plot scatter + line + ±3 SEM band.

    The slope and r of the least-squares fit are kept as *descriptive* effect
    sizes only. Significance comes from ``mantel`` — a :func:`mantel_test`
    result dict — because the pairs are not independent (each cell appears in
    many pairs); the ordinary-regression p-value would be badly anti-
    conservative. When ``mantel`` is None the line is labelled descriptive-only.

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
                zorder=1, rasterized=True,
            )
        return False
    xv, yv = dists[valid], corrs[valid]
    res = linregress(xv, yv)
    x_line = np.linspace(xv.min(), xv.max(), 100)
    y_line = res.slope * x_line + res.intercept

    ax.scatter(
        xv, yv,
        color=color, alpha=0.30, s=8, edgecolors="none",
        zorder=1, rasterized=True,
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

    if mantel is None:
        stat_line = "descriptive fit only (no Mantel test)"
    elif mantel.get("insufficient"):
        stat_line = f"Mantel: n={mantel.get('n_cells', 0)} cells — too few"
    else:
        stat_line = (
            f"Mantel p={mantel['p_value']:.3g}  "
            f"({mantel['n_perm']} perms, {mantel['n_cells']} cells)"
        )

    ax.plot(
        x_line, y_line,
        color=color,
        linewidth=PLOT_PARAMS["mean_lw"],
        label=(
            f"{label_prefix} (n={n} pairs)\n"
            f"slope={res.slope:.2e}  r={res.rvalue:.3f}  (descriptive)\n"
            f"{stat_line}"
        ),
        zorder=3,
    )
    return True


def _scatter_corr_vs_dist(
    ax, pw_dist, pw_corr, *, pair_classes=None,
    corr_method="pearson", title="", mantel_all=None, mantel_rr=None,
):
    """Scatter pairwise (distance, correlation), optionally split by pair class.

    When ``pair_classes`` is None or all-empty, falls back to a single fit.
    ``mantel_all`` / ``mantel_rr`` are :func:`mantel_test` results for the
    full cell set and the responder sub-matrix respectively, used to label
    each fit with a pseudoreplication-safe p-value.
    """
    ax.spines[["top", "right"]].set_visible(False)
    drew_any = False
    if pair_classes and any(m.any() for m in pair_classes.values()):
        # Non-responder (NN) pairs: gray scatter only, no fit line.
        nn_mask = pair_classes.get("NN")
        if nn_mask is not None and nn_mask.any():
            valid = nn_mask & ~np.isnan(pw_dist) & ~np.isnan(pw_corr)
            n_nn = int(valid.sum())
            if n_nn > 0:
                ax.scatter(
                    pw_dist[valid], pw_corr[valid],
                    color=PAIR_CLASS_COLORS["NN"], alpha=0.30, s=8,
                    edgecolors="none", zorder=1, rasterized=True,
                    label=f"{PAIR_CLASS_LABEL['NN']} (n={n_nn})",
                )
                drew_any = True
        # Responder × responder (RR) pairs: blue scatter + fit line + band.
        rr_mask = pair_classes.get("RR")
        if rr_mask is not None and rr_mask.any():
            drew_any |= _fit_and_plot_subset(
                ax,
                pw_dist[rr_mask], pw_corr[rr_mask],
                color=PAIR_CLASS_COLORS["RR"],
                label_prefix=PAIR_CLASS_LABEL["RR"],
                mantel=mantel_rr,
            )
    else:
        drew_any = _fit_and_plot_subset(
            ax,
            np.asarray(pw_dist, dtype=float),
            np.asarray(pw_corr, dtype=float),
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All pairs",
            mantel=mantel_all,
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
        for entry in per_channel_pairs:
            pw_dist = entry["pw_dist"]
            pw_corr = entry["pw_corr_by_method"][method]
            pair_classes = entry.get("pair_classes")

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
            # Non-responder (NN) pairs: gray scatter only, no fit line.
            nn_mask = merged_classes.get("NN")
            if nn_mask is not None and nn_mask.any():
                valid = nn_mask & ~np.isnan(dists) & ~np.isnan(corrs)
                n_nn = int(valid.sum())
                if n_nn > 0:
                    ax.scatter(
                        dists[valid], corrs[valid],
                        color=PAIR_CLASS_COLORS["NN"], alpha=0.20, s=6,
                        edgecolors="none", zorder=1, rasterized=True,
                        label=f"{PAIR_CLASS_LABEL['NN']} (pooled, n={n_nn})",
                    )
            # Responder × responder (RR) pairs: blue scatter + fit line + band.
            rr_mask = merged_classes.get("RR")
            if rr_mask is not None and rr_mask.any():
                _fit_and_plot_subset(
                    ax,
                    dists[rr_mask], corrs[rr_mask],
                    color=PAIR_CLASS_COLORS["RR"],
                    label_prefix=f"{PAIR_CLASS_LABEL['RR']} (pooled)",
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
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="lower right")

        # A single pooled Mantel test would mix independent dishes and has no
        # valid label permutation. Instead each channel gets its own cell-level
        # Mantel test, and the per-replicate Mantel r values are combined with a
        # one-sample t-test against 0 — the replicate-level biological test the
        # rest of the pipeline treats as the unit of inference.
        mantel_lines = []
        r_per_channel = []
        for entry in per_channel_pairs:
            res = entry.get("mantel", {}).get(method, {}).get("all")
            if res and not res.get("insufficient"):
                mantel_lines.append(
                    f"{entry['ch']}: r={res['r_obs']:+.3f}, p={res['p_value']:.3g}"
                )
                r_per_channel.append(res["r_obs"])
        if mantel_lines:
            combined = one_sample_t_dz(r_per_channel)
            if combined["n"] >= 2 and np.isfinite(combined["p_value"]):
                mantel_lines.append(
                    f"combined ({combined['n']} replicates): "
                    f"mean r={combined['mean']:+.3f}, "
                    f"t={combined['t_stat']:+.2f}, p={combined['p_value']:.3g}, "
                    f"dz={combined['cohen_dz']:+.2f}"
                )
            else:
                mantel_lines.append(
                    f"combined: n={combined['n']} replicate(s) — too few to test"
                )
            ax.text(
                0.02, 0.97,
                "Mantel test (cell-level per channel; combined =\n"
                "one-sample t of per-channel Mantel r vs 0):\n"
                + "\n".join(mantel_lines),
                ha="left", va="top", transform=ax.transAxes,
                fontsize=PLOT_PARAMS["legend_fontsize"] - 1,
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
            )

    n_channels = len(per_channel_pairs)
    plt.tight_layout(rect=(0, 0.03, 1, 1));
    fig.text(
        0.5, 0.006,
        inferential_caveat(
            exp_name, n_channels, unit="cell pair",
            extra="Pooled line is descriptive; significance = per-channel "
                  "Mantel test combined across replicates by a one-sample "
                  "t-test of the per-channel Mantel r against 0.",
        ),
        ha="center", va="bottom",
        fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
        style="italic", color="#555555",
    )
    save_fig(
        fig, fig_path(exp_name, "corr_vs_dist_combined"),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)


def plot_correlation_vs_distance(experiments, state):
    """Per experiment: Pearson + Spearman correlation vs pairwise distance.

    Two-row figure with Pearson on row 0 and Spearman on row 1, one column
    per channel, computed over the full corrected time series. Pairs are
    colored by responder-pair class when responder thresholds are available.
    """
    responder_masks = compute_responder_masks(experiments, state)
    window_label = "full corrected time series"

    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        # Build per-channel context.
        per_channel_ctx = {}
        for ch in channels:
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
            pos_xy = np.array(pos_xy, dtype=float)

            if len(pos_xy) < 2:
                per_channel_ctx[ch] = None
                continue

            mat_k = mat[keep_rows]
            dist_mat = squareform(pdist(pos_xy, metric="euclidean"))
            dist_sq = dist_mat / PIXELS_PER_UM
            iu = np.triu_indices(len(pos_xy), k=1)
            pw_dist = dist_sq[iu]

            pair_classes = None
            row_mask = None
            # ``compute_responder_masks`` rows follow the corrected-lum
            # CellID order, i.e. the same order as ``cell_ids_int``.
            full_mask = responder_masks.get((exp_name, ch))
            if full_mask is not None:
                row_mask = np.array(
                    [bool(full_mask[r]) for r in keep_rows],
                    dtype=bool,
                )
                pair_classes = _classify_pair_classes(row_mask)

            per_channel_ctx[ch] = {
                "mat_k": mat_k,
                "pw_dist": pw_dist,
                "dist_sq": dist_sq,
                "iu": iu,
                "n_cells": len(keep_rows),
                "pair_classes": pair_classes,
                "row_mask": row_mask,
            }

        fig, axes = plt.subplots(
            2, len(channels),
            figsize=(6 * len(channels), 11),
            dpi=PLOT_PARAMS["dpi"], sharey="row",
        )
        if len(channels) == 1:
            axes = axes.reshape(2, 1)

        per_channel_pairs = []

        for col, ch in enumerate(channels):
            ctx = per_channel_ctx.get(ch)
            if ctx is None:
                for row in range(2):
                    axes[row, col].set_title(
                        f"{ch}: insufficient data",
                        fontsize=PLOT_PARAMS["title_fontsize"],
                    )
                continue

            mat_k = ctx["mat_k"]
            if mat_k.shape[1] < MIN_FRAMES_FOR_CORR:
                for row in range(2):
                    axes[row, col].set_title(
                        f"{ch}: insufficient samples",
                        fontsize=PLOT_PARAMS["title_fontsize"],
                    )
                continue

            pearson_mat = pd.DataFrame(mat_k).T.corr(method="pearson").values
            spearman_mat = pd.DataFrame(mat_k).T.corr(method="spearman").values
            iu = ctx["iu"]
            pw_corr_by_method = {
                "pearson": pearson_mat[iu],
                "spearman": spearman_mat[iu],
            }

            # Mantel test per correlation method. Permuting cell labels makes
            # the *cell* (not the pair) the unit of exchangeability, which is
            # the pseudoreplication-safe significance test here. The responder
            # sub-matrix gets its own Mantel when ≥4 responder cells exist.
            dist_sq = ctx["dist_sq"]
            row_mask = ctx["row_mask"]
            mantel_by_method = {}
            for mth, cmat in (("pearson", pearson_mat),
                              ("spearman", spearman_mat)):
                m_all = mantel_test(dist_sq, cmat, n_perm=999)
                m_rr = None
                if row_mask is not None and int(row_mask.sum()) >= 4:
                    rr_idx = np.where(row_mask)[0]
                    m_rr = mantel_test(
                        dist_sq[np.ix_(rr_idx, rr_idx)],
                        cmat[np.ix_(rr_idx, rr_idx)], n_perm=999,
                    )
                mantel_by_method[mth] = {"all": m_all, "RR": m_rr}

            per_channel_pairs.append({
                "ch": ch,
                "n_cells": ctx["n_cells"],
                "pw_dist": ctx["pw_dist"],
                "pw_corr_by_method": pw_corr_by_method,
                "pair_classes": ctx["pair_classes"],
                "mantel": mantel_by_method,
            })

            for row, method in enumerate(("pearson", "spearman")):
                _scatter_corr_vs_dist(
                    axes[row, col],
                    ctx["pw_dist"], pw_corr_by_method[method],
                    pair_classes=ctx["pair_classes"],
                    corr_method=method,
                    title=f"{ch}  ({ctx['n_cells']} cells, {METHOD_LABEL[method]})",
                    mantel_all=mantel_by_method[method]["all"],
                    mantel_rr=mantel_by_method[method]["RR"],
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

        fig.suptitle(
            f"{exp_name} — pairwise correlation vs pairwise distance "
            f"({window_label}, Pearson top, Spearman bottom)",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold", y=1.01,
        )
        plt.tight_layout(rect=(0, 0.03, 1, 1));
        fig.text(
            0.5, 0.006,
            inferential_caveat(
                exp_name, len(channels), unit="cell pair",
                extra="Significance: Mantel permutation test (per channel); "
                      "slope/r are descriptive.",
            ),
            ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
        save_fig(
            fig, fig_path(exp_name, "corr_vs_dist"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)

        _plot_corr_vs_dist_combined(
            exp_name, per_channel_pairs, window_label=window_label,
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_correlation_vs_distance(experiments, state)


if __name__ == "__main__":
    main()
