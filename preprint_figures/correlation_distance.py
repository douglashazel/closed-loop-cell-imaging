#!/usr/bin/env python3
"""Pairwise correlation vs spatial distance (full time series).

Per experiment:
    * corr_vs_dist.png           — per-channel scatter panels
    * corr_vs_dist_combined.png  — pooled across channels

Both use the full dF/F0-normalized time series (per-cell baseline-normalized
to keep the metric consistent with the rest of the pipeline). Within each
panel, pairs are coloured by responder-pair status (RR / NN / RN) when
responder thresholds are available, with a separate descriptive regression
line per subset. Falls back to a single-cloud scatter when no responder
threshold applies for the channel. Significance is a Mantel permutation test
(per channel) — the regression p-value is invalid here because the ~N²/2
cell pairs are not independent. Distances are reported in μm using the
imaging calibration ``PIXELS_PER_UM``.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
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
from common.stim_helpers import compute_f0_baseline
from figstyle import apply_style

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402

apply_style()


PIXELS_PER_UM = 180.1  # imaging calibration: 0.00555 μm/pixel
MIN_FRAMES_FOR_CORR = 5
PAIR_CLASS_COLORS = {
    "RR": PLOT_PARAMS["rr_color"],   # blue — both responders
    "NN": "#7f7f7f",                 # gray — both non-responders
}
PAIR_CLASS_LABEL = {
    "RR": "Responders-only",
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


def _fit_and_plot_subset(ax, dists, corrs, color, label_prefix, *, mantel=None,
                         draw_scatter=True, scatter_label=None, stat_text=None,
                         legend_label=None):
    """Fit a line on (dists, corrs); plot scatter + line + ±3 SEM band.

    The slope and r of the least-squares fit are kept as *descriptive* effect
    sizes only. Significance comes from ``mantel`` — a :func:`mantel_test`
    result dict — because the pairs are not independent (each cell appears in
    many pairs); the ordinary-regression p-value would be badly anti-
    conservative. When ``mantel`` is None the line is labelled descriptive-only;
    ``stat_text`` overrides the derived stat line verbatim when given.

    ``draw_scatter`` plots the point cloud (set False to add a fit line over an
    already-drawn cloud); ``scatter_label`` gives that cloud its own legend
    entry (e.g. the blue responder dot, mirroring the gray non-responder dot).
    ``legend_label`` overrides the entire fit-line legend entry verbatim — used
    for the clean, caption-oriented Pearson legend (a short "Responders-only
    (Mantel p = …)" in place of the multi-line descriptive slope/r block).

    Returns True if a line was drawn, False if too few valid points.
    """
    valid = ~np.isnan(dists) & ~np.isnan(corrs)
    n = int(valid.sum())
    if n < 3:
        if n > 0 and draw_scatter:
            ax.scatter(
                dists[valid], corrs[valid],
                color=color, alpha=0.35, s=8, edgecolors="none",
                label=f"{scatter_label or label_prefix} (n={n})",
                zorder=1, rasterized=True,
            )
        return False
    xv, yv = dists[valid], corrs[valid]
    res = linregress(xv, yv)
    x_line = np.linspace(xv.min(), xv.max(), 100)
    y_line = res.slope * x_line + res.intercept

    if draw_scatter:
        ax.scatter(
            xv, yv,
            color=color, alpha=0.30, s=8, edgecolors="none",
            label=f"{scatter_label} (n={n})" if scatter_label else None,
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

    if stat_text is not None:
        stat_line = stat_text
    elif mantel is None:
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
            legend_label if legend_label is not None else
            f"{label_prefix} (n={n} pairs)\n"
            f"slope={res.slope:.2e}  r={res.rvalue:.3f}  (descriptive)\n"
            f"{stat_line}"
        ),
        zorder=3,
    )
    return True


def _combined_mantel_stat_text(per_channel_pairs, method):
    """One-line Mantel summary for the pooled fit's legend.

    Each channel contributes its own cell-level Mantel r; those per-replicate
    r values are combined with a one-sample t-test against 0 — the replicate-
    level biological test. A single pooled Mantel test would mix independent
    dishes and has no valid label permutation, so it is not used.
    """
    r_per_channel = []
    for entry in per_channel_pairs:
        res = entry.get("mantel", {}).get(method, {}).get("all")
        if res and not res.get("insufficient"):
            r_per_channel.append(res["r_obs"])
    if not r_per_channel:
        return "Mantel: no channel had enough cells to test"
    if len(r_per_channel) < 2:
        return (
            f"Mantel r={r_per_channel[0]:+.3f} "
            f"(1 replicate — no replicate-level test)"
        )
    combined = one_sample_t_dz(r_per_channel)
    if np.isfinite(combined["p_value"]):
        return (
            f"Mantel (per-channel r vs 0, {combined['n']} replicates): "
            f"mean r={combined['mean']:+.3f}, p={combined['p_value']:.3g}, "
            f"dz={combined['cohen_dz']:+.2f}"
        )
    return f"Mantel: {combined['n']} replicate(s) — too few to test"


def _combined_mantel_pvalue(per_channel_pairs, method, key):
    """Replicate-level Mantel result for one subset (``key`` = "all" or "RR").

    Combines each channel's cell-level Mantel r for ``key`` via a one-sample
    t-test of the per-channel r against 0 — the biological-replicate test that
    underlies the pooled fit line (the ordinary regression p over the pooled
    pairs is invalid; the pairs are not independent). Returns
    ``{"p_value", "mean_r", "n"}`` (``p_value`` is NaN when only one channel was
    testable) or ``None`` when no channel had enough cells.
    """
    r_per_channel = []
    for entry in per_channel_pairs:
        res = entry.get("mantel", {}).get(method, {}).get(key)
        if res and not res.get("insufficient"):
            r_per_channel.append(res["r_obs"])
    if not r_per_channel:
        return None
    if len(r_per_channel) < 2:
        return {"p_value": np.nan, "mean_r": float(r_per_channel[0]), "n": 1}
    combined = one_sample_t_dz(r_per_channel)
    return {
        "p_value": combined["p_value"],
        "mean_r": combined["mean"],
        "n": combined["n"],
    }


def _fmt_p(p):
    """Compact p-value string for a legend (sci-notation below 1e-3)."""
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}"


def _mantel_line_label(base, stat):
    """``base`` plus a replicate-level Mantel p suffix when one is available."""
    if stat is None or not np.isfinite(stat.get("p_value", np.nan)):
        return base
    return f"{base} (Mantel p = {_fmt_p(stat['p_value'])})"


def _apply_log1p_xaxis(axes, xlabel="Pairwise distance (μm, log1p axis)"):
    """Switch each axis to a log1p display scale — visualization only.

    Underlying data and fit lines are untouched; only the x-axis transform
    changes, so linear fits drawn over a log1p axis will display as gently
    curved (which is the honest depiction of a linear-in-distance fit on a
    log1p scale). Explicit ticks at biologically interpretable distances
    keep the short-range region readable instead of letting matplotlib
    place ticks at evenly-spaced transformed values.
    """
    candidate_ticks = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    for ax in np.atleast_1d(axes).flat:
        lo, hi = ax.get_xlim()
        if hi <= 0:
            continue
        ax.set_xlim(max(0.0, lo), hi)
        ax.set_xscale("function", functions=(np.log1p, np.expm1))
        new_lo, new_hi = ax.get_xlim()
        visible = [t for t in candidate_ticks if new_lo <= t <= new_hi]
        if len(visible) >= 2:
            ax.xaxis.set_major_locator(FixedLocator(visible))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if ax.get_xlabel():
            ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])


def _scatter_corr_vs_dist(
    ax, pw_dist, pw_corr, *, pair_classes=None,
    corr_method="pearson", title="", mantel_all=None, mantel_rr=None,
    clean_legend=False,
):
    """Scatter pairwise (distance, correlation), optionally split by pair class.

    When ``pair_classes`` is None or all-empty, falls back to a single fit.
    ``mantel_all`` / ``mantel_rr`` are :func:`mantel_test` results for the
    full cell set and the responder sub-matrix respectively, used to label
    each fit with a pseudoreplication-safe p-value.

    ``clean_legend`` switches to the caption-oriented legend used for the NRK
    per-chamber Pearson panels: the scatter clouds drop their own legend
    entries and the two fit lines are labelled ``"Responders-only (Mantel
    p = …)"`` and ``"All cells (Mantel p = …)"`` from this chamber's own
    per-channel Mantel test. The descriptive slope/r/pair-count move to the
    written caption. Mirrors the pooled (combined) figure's Pearson legend,
    one panel per chamber.
    """
    ax.spines[["top", "right"]].set_visible(False)
    drew_any = False
    if pair_classes and any(m.any() for m in pair_classes.values()):
        # Non-responder (NN) pairs: gray scatter only, no fit line. The clean
        # legend drops the NN entry — the two coloured fit lines carry meaning.
        nn_mask = pair_classes.get("NN")
        if nn_mask is not None and nn_mask.any():
            valid = nn_mask & ~np.isnan(pw_dist) & ~np.isnan(pw_corr)
            n_nn = int(valid.sum())
            if n_nn > 0:
                ax.scatter(
                    pw_dist[valid], pw_corr[valid],
                    color=PAIR_CLASS_COLORS["NN"], alpha=0.30, s=8,
                    edgecolors="none", zorder=1, rasterized=True,
                    label=(None if clean_legend
                           else f"{PAIR_CLASS_LABEL['NN']} (n={n_nn})"),
                )
                drew_any = True
        # Responder × responder (RR) pairs: blue scatter + fit line + band. The
        # clean legend suppresses the blue scatter entry and labels the fit line
        # "Responders-only (Mantel p = …)".
        rr_mask = pair_classes.get("RR")
        if rr_mask is not None and rr_mask.any():
            drew_any |= _fit_and_plot_subset(
                ax,
                pw_dist[rr_mask], pw_corr[rr_mask],
                color=PAIR_CLASS_COLORS["RR"],
                label_prefix=PAIR_CLASS_LABEL["RR"],
                scatter_label=(None if clean_legend
                               else PAIR_CLASS_LABEL["RR"]),
                legend_label=(_mantel_line_label(PAIR_CLASS_LABEL["RR"], mantel_rr)
                              if clean_legend else None),
                # mantel=mantel_rr,
            )
        # All cells pooled (RR + NN + RN): one fit line over every pair, drawn
        # over the already-coloured clouds without redrawing any scatter. The
        # clean legend labels it "All cells (Mantel p = …)".
        drew_any |= _fit_and_plot_subset(
            ax,
            np.asarray(pw_dist, dtype=float),
            np.asarray(pw_corr, dtype=float),
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All cells",
            legend_label=(_mantel_line_label("All cells", mantel_all)
                          if clean_legend else None),
            # mantel=mantel_all,
            draw_scatter=False,
        )
    else:
        drew_any = _fit_and_plot_subset(
            ax,
            np.asarray(pw_dist, dtype=float),
            np.asarray(pw_corr, dtype=float),
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All pairs",
            # mantel=mantel_all,
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

    fig, axes = plt.subplots(
        2, 1, figsize=(PLOT_PARAMS["width_full"], 8.5), dpi=PLOT_PARAMS["dpi"],
    )

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

        # Mantel summary for the pooled fit's legend (replaces the old top-left
        # stats box) — see _combined_mantel_stat_text for why it is per-channel.
        mantel_stat = _combined_mantel_stat_text(per_channel_pairs, method)

        # The Pearson (top) panel gets a clean, caption-oriented legend: each
        # fit line is labelled with its replicate-level Mantel p (the valid
        # significance test), with the descriptive slope/r/n moved to the
        # written caption. The Spearman (bottom) panel keeps the verbose
        # descriptive legend unchanged.
        is_pearson = method == "pearson"
        p_rr = (_combined_mantel_pvalue(per_channel_pairs, method, "RR")
                if is_pearson else None)
        p_all = (_combined_mantel_pvalue(per_channel_pairs, method, "all")
                 if is_pearson else None)

        if merged_classes is not None and any(m.any() for m in merged_classes.values()):
            # Non-responder (NN) pairs: gray scatter only, no fit line. On the
            # Pearson panel the gray dots drop their own legend entry too — the
            # coloured fit lines signal what each scatter colour means.
            nn_mask = merged_classes.get("NN")
            if nn_mask is not None and nn_mask.any():
                valid = nn_mask & ~np.isnan(dists) & ~np.isnan(corrs)
                n_nn = int(valid.sum())
                if n_nn > 0:
                    ax.scatter(
                        dists[valid], corrs[valid],
                        color=PAIR_CLASS_COLORS["NN"], alpha=0.20, s=6,
                        edgecolors="none", zorder=1, rasterized=True,
                        label=(None if is_pearson
                               else f"{PAIR_CLASS_LABEL['NN']}"),
                    )
            # Responder × responder (RR) pairs: blue scatter + fit line + band.
            # On the Pearson panel the blue dots drop their own legend entry so
            # the single "Responders-only (Mantel p = …)" line speaks for them.
            rr_mask = merged_classes.get("RR")
            if rr_mask is not None and rr_mask.any():
                _fit_and_plot_subset(
                    ax,
                    dists[rr_mask], corrs[rr_mask],
                    color=PAIR_CLASS_COLORS["RR"],
                    label_prefix=f"{PAIR_CLASS_LABEL['RR']} (pooled)",
                    scatter_label=(None if is_pearson
                                   else f"{PAIR_CLASS_LABEL['RR']} (pooled)"),
                    legend_label=(_mantel_line_label(PAIR_CLASS_LABEL["RR"], p_rr)
                                  if is_pearson else None),
                )
            # All cells pooled (RR + NN + RN): one fit line over every pair,
            # drawn over the already-coloured clouds (no extra scatter).
            _fit_and_plot_subset(
                ax,
                dists, corrs,
                color=PLOT_PARAMS["corr_fit_color"],
                label_prefix="All cells pooled",
                # stat_text=mantel_stat,
                draw_scatter=False,
                legend_label=(_mantel_line_label("All cells", p_all)
                              if is_pearson else None),
            )
        else:
            _fit_and_plot_subset(
                ax,
                dists, corrs,
                color=PLOT_PARAMS["corr_fit_color"],
                label_prefix="Pooled",
                # stat_text=mantel_stat,
            )

        ax.set_xlabel(
            "Pairwise distance (μm)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_ylabel(
            f"{METHOD_LABEL[method]} ({window_label})",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        if is_pearson:
            ax.set_title(
                "Pairwise Pearson r vs. distance",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
        else:
            ax.set_title(
                f"{exp_name} — pairwise {METHOD_LABEL[method]} vs distance "
                f"({window_label}, all channels combined)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="lower right")

    n_channels = len(per_channel_pairs)
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
        dpi=PLOT_PARAMS["dpi"],
    )

    _apply_log1p_xaxis(axes)
    for ax in np.atleast_1d(axes).flat:
        ttl = ax.get_title()
        if ttl:
            ax.set_title(
                f"{ttl}  [log1p distance axis]",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
    save_fig(
        fig, fig_path(exp_name, "corr_vs_dist_combined_log1p"),
        dpi=PLOT_PARAMS["dpi"],
    )
    plt.close(fig)


def plot_correlation_vs_distance(experiments, state):
    """Per experiment: Pearson + Spearman correlation vs pairwise distance.

    Two-row figure with Pearson on row 0 and Spearman on row 1, one column
    per channel, computed over the full dF/F0-normalized time series. Pairs
    are colored by responder-pair class when responder thresholds are
    available.
    """
    responder_masks = compute_responder_masks(experiments, state)
    window_label = "full dF/F0 time series"

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
            # Normalize each cell to its own pre-stim baseline so the
            # correlation is computed on dF/F0 (consistent with the rest of
            # the pipeline — responders, time traces, etc.). F0 rows align
            # with df.index because compute_f0_baseline reads the same
            # corrected_lum table.
            F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            mat = (mat - F0) / F0_safe
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

        # Channels are stacked as ROWS (the two correlation methods are the
        # columns) so the figure fits the locked single-column width (6.5 in)
        # for any channel count. A 2-row x N-channel layout collapses
        # horizontally past ~3 channels; height instead grows with channels.
        fig, axes = plt.subplots(
            len(channels), 2,
            figsize=(PLOT_PARAMS["width_full"], 2.7 * len(channels) + 1.0),
            dpi=PLOT_PARAMS["dpi"], sharey="col",
        )
        axes = np.asarray(axes).reshape(len(channels), 2)

        per_channel_pairs = []

        for row, ch in enumerate(channels):
            ctx = per_channel_ctx.get(ch)
            if ctx is None:
                for col in range(2):
                    axes[row, col].set_title(
                        f"{ch}: insufficient data",
                        fontsize=PLOT_PARAMS["title_fontsize"],
                    )
                continue

            mat_k = ctx["mat_k"]
            if mat_k.shape[1] < MIN_FRAMES_FOR_CORR:
                for col in range(2):
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

            # NRK encodes the chamber as a trailing letter ("channel 1 A");
            # C2C12/PC3 channels end in a digit. Only the chamber-labelled
            # (NRK) Pearson panels get the caption-oriented title + clean
            # "Responders-only / All cells (Mantel p = …)" legend; every other
            # panel keeps the verbose descriptive legend unchanged.
            chamber = ch.split()[-1]
            is_chamber = chamber.isalpha()
            for col, method in enumerate(("pearson", "spearman")):
                clean = is_chamber and method == "pearson"
                if clean:
                    panel_title = (
                        f"Pairwise Pearson r vs. distance: chamber {chamber}"
                    )
                else:
                    panel_title = (
                        f"{ch}  ({ctx['n_cells']} cells, {METHOD_LABEL[method]})"
                    )
                _scatter_corr_vs_dist(
                    axes[row, col],
                    ctx["pw_dist"], pw_corr_by_method[method],
                    pair_classes=ctx["pair_classes"],
                    corr_method=method,
                    title=panel_title,
                    mantel_all=mantel_by_method[method]["all"],
                    mantel_rr=mantel_by_method[method]["RR"],
                    clean_legend=clean,
                )
                axes[row, col].set_xlabel(
                    "Pairwise distance (μm)",
                    fontsize=PLOT_PARAMS["axis_label_fontsize"],
                )
                if col == 0:
                    # Col 0 is Pearson. The clean chamber panels name the metric
                    # (the chamber is in the title); otherwise the row is
                    # labelled by channel (the method lives in each title).
                    axes[row, col].set_ylabel(
                        f"Pearson r ({window_label})" if clean
                        else f"{ch} ({window_label})",
                        fontsize=PLOT_PARAMS["axis_label_fontsize"],
                    )

        fig.suptitle(
            f"{exp_name} — pairwise correlation vs pairwise distance "
            f"({window_label}, Pearson left, Spearman right)",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold",
        )
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
            dpi=PLOT_PARAMS["dpi"],
        )

        _apply_log1p_xaxis(axes)
        fig.suptitle(
            f"{exp_name} — pairwise correlation vs pairwise distance "
            f"({window_label}, Pearson left, Spearman right)  "
            f"[log1p distance axis]",
            fontsize=PLOT_PARAMS["title_fontsize"] + 1,
            fontweight="bold",
        )
        save_fig(
            fig, fig_path(exp_name, "corr_vs_dist_log1p"),
            dpi=PLOT_PARAMS["dpi"],
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
