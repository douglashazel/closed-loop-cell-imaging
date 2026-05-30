"""Render functions for the correlation-vs-distance figures
(reads analysis_cache/<exp>/correlation_distance.pkl).

The old ``n_channels x 2`` grid (Pearson left, Spearman right) and the 2-row
pooled ``corr_vs_dist_combined`` figure are decomposed into STANDALONE
single-axis figures:

  * per (channel, method): ``{ch}_corr_vs_dist_pearson`` /
    ``{ch}_corr_vs_dist_spearman``
  * per method, pooled across channels: ``corr_vs_dist_combined_pearson`` /
    ``corr_vs_dist_combined_spearman``

Every figure also gets a ``_log1p`` twin (a render-time ``apply_log1p`` flag in
``fill``) that switches the x display to a log1p scale WITHOUT changing the data.

The descriptive least-squares fit line (``scipy.linregress``) is recomputed HERE
from the cached ``pw_dist`` / ``pw_corr`` vectors — it is deterministic. The
Mantel p-values come from the cache (the analysis layer ran the permutation
tests); this layer never recomputes a significance test. Label / title / caveat
text lives centrally in ``figures_spec.py``; these functions pull it from the
passed ``spec``.
"""
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.stats import linregress

from plots._base import (
    PLOT_PARAMS,
    clean_axes,
    title_of,
    xlabel_of,
    ylabel_of,
)

NAME = "correlation_distance"

PAIR_CLASS_COLORS = {
    "RR": PLOT_PARAMS["rr_color"],   # blue — both responders
    "NN": "#7f7f7f",                 # gray — both non-responders
}
PAIR_CLASS_LABEL = {
    "RR": "Responders-only",
    "NN": "Non-responder pairs",
}
METHOD_LABEL = {"pearson": "Pearson r", "spearman": "Spearman ρ"}


# =============================================================================
# Small label helpers (ported verbatim from the source).
# =============================================================================
def _fmt_p(p):
    """Compact p-value string for a legend (sci-notation below 1e-3)."""
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}"


def _mantel_line_label(base, stat):
    """``base`` plus a replicate-level Mantel p suffix when one is available."""
    if stat is None or not np.isfinite(stat.get("p_value", np.nan)):
        return base
    return f"{base} (Mantel p = {_fmt_p(stat['p_value'])})"


def _apply_log1p_xaxis(ax, xlabel="Pairwise distance (μm, log1p axis)"):
    """Switch the axis to a log1p display scale — visualization only.

    Underlying data and fit lines are untouched; only the x-axis transform
    changes, so linear fits drawn over a log1p axis display as gently curved
    (the honest depiction of a linear-in-distance fit on a log1p scale).
    Explicit ticks at biologically interpretable distances keep the short-range
    region readable.
    """
    candidate_ticks = [0, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    lo, hi = ax.get_xlim()
    if hi <= 0:
        return
    ax.set_xlim(max(0.0, lo), hi)
    ax.set_xscale("function", functions=(np.log1p, np.expm1))
    new_lo, new_hi = ax.get_xlim()
    visible = [t for t in candidate_ticks if new_lo <= t <= new_hi]
    if len(visible) >= 2:
        ax.xaxis.set_major_locator(FixedLocator(visible))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])


# =============================================================================
# Descriptive fit + scatter (recomputed from cached vectors).
# =============================================================================
def _fit_and_plot_subset(ax, dists, corrs, color, label_prefix, *,
                         draw_scatter=True, scatter_label=None, stat_text=None,
                         legend_label=None):
    """Fit a line on (dists, corrs); plot scatter + line + ±3 SEM band.

    The slope/r of the least-squares fit are kept as *descriptive* effect sizes
    only; significance lives in the cached Mantel p (passed via ``legend_label``
    / ``stat_text``). Returns True if a line was drawn, False otherwise. Ported
    verbatim from the source (the ``mantel`` argument was already dead there).
    """
    dists = np.asarray(dists, dtype=float)
    corrs = np.asarray(corrs, dtype=float)
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
    else:
        stat_line = "descriptive fit only (no Mantel test)"

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


def _scatter_corr_vs_dist(ax, pw_dist, pw_corr, *, pair_classes=None,
                          p_all=None, p_rr=None, clean_legend=False):
    """Scatter pairwise (distance, correlation), optionally split by pair class.

    When ``pair_classes`` is None or all-empty, falls back to a single fit.
    ``clean_legend`` switches to the caption-oriented Pearson legend used for the
    NRK per-chamber panels: the scatter clouds drop their own legend entries and
    the two fit lines are labelled "Responders-only (Mantel p = …)" / "All cells
    (Mantel p = …)" from this channel's cached per-cell Mantel result. Returns
    True if anything was drawn (the caller adds the legend).
    """
    drew_any = False
    if pair_classes and any(np.asarray(m).any() for m in pair_classes.values()):
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
        # Responder × responder (RR) pairs: blue scatter + fit line + band.
        rr_mask = pair_classes.get("RR")
        if rr_mask is not None and rr_mask.any():
            drew_any |= _fit_and_plot_subset(
                ax,
                pw_dist[rr_mask], pw_corr[rr_mask],
                color=PAIR_CLASS_COLORS["RR"],
                label_prefix=PAIR_CLASS_LABEL["RR"],
                scatter_label=(None if clean_legend
                               else PAIR_CLASS_LABEL["RR"]),
                legend_label=(_mantel_line_label(PAIR_CLASS_LABEL["RR"], p_rr)
                              if clean_legend else None),
            )
        # All cells pooled (RR + NN + RN): one fit line over every pair, drawn
        # over the already-coloured clouds without redrawing any scatter.
        drew_any |= _fit_and_plot_subset(
            ax,
            pw_dist, pw_corr,
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All cells",
            legend_label=(_mantel_line_label("All cells", p_all)
                          if clean_legend else None),
            draw_scatter=False,
        )
    else:
        drew_any = _fit_and_plot_subset(
            ax,
            pw_dist, pw_corr,
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All pairs",
        )
    return drew_any


# =============================================================================
# Render functions (one figure each).
# =============================================================================
def render_corr_vs_dist_channel(ax, payload, spec, *, fill):
    """One per-channel panel: pairwise correlation vs distance for one method.

    ``payload`` carries this channel's cached ``pw_dist`` / ``pw_corr`` vectors,
    ``pair_classes`` and the per-cell Mantel results (``mantel_all`` /
    ``mantel_rr``). The clean Pearson legend uses those Mantel p-values; the
    Spearman / non-chamber panels keep the verbose descriptive legend.
    """
    P = PLOT_PARAMS
    clean_axes(ax)

    pw_dist = np.asarray(payload["pw_dist"], dtype=float)
    pw_corr = np.asarray(payload["pw_corr"], dtype=float)
    pair_classes = payload.get("pair_classes")
    clean = bool(payload.get("clean_legend"))

    # Cached per-cell Mantel results → the clean Pearson legend's p-values. The
    # source labelled the line from ``mantel.get("p_value")`` via _mantel_line_
    # label; mirror that with a {"p_value": ...} dict.
    m_all = payload.get("mantel_all")
    m_rr = payload.get("mantel_rr")
    p_all = ({"p_value": m_all.get("p_value")}
             if (m_all and not m_all.get("insufficient")) else None)
    p_rr = ({"p_value": m_rr.get("p_value")}
            if (m_rr and not m_rr.get("insufficient")) else None)

    drew_any = _scatter_corr_vs_dist(
        ax, pw_dist, pw_corr,
        pair_classes=pair_classes,
        p_all=p_all, p_rr=p_rr, clean_legend=clean,
    )

    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    # The "[log1p distance axis]" title suffix is baked into the template via
    # {log1p_note}; the log1p switch only restyles the x-axis (label + ticks).
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])

    if fill.get("apply_log1p"):
        _apply_log1p_xaxis(ax)

    if drew_any:
        ax.legend(fontsize=P["legend_fontsize"], loc="best");


def render_corr_vs_dist_combined(ax, payload, spec, *, fill):
    """One pooled panel: pairwise correlation vs distance, all channels combined.

    ``payload`` carries the per-channel ``pw_dist`` / ``pw_corr`` / ``pair_class``
    lists already concatenated here, plus the cached replicate-level Mantel
    results (``p_all`` / ``p_rr``) for the clean Pearson legend.
    """
    P = PLOT_PARAMS
    clean_axes(ax)

    dists = np.asarray(payload["pw_dist"], dtype=float)
    corrs = np.asarray(payload["pw_corr"], dtype=float)
    merged_classes = payload.get("pair_classes")  # dict of arrays or None
    is_pearson = payload["method"] == "pearson"
    p_rr = payload.get("p_rr") if is_pearson else None
    p_all = payload.get("p_all") if is_pearson else None

    if merged_classes is not None and any(
        np.asarray(m).any() for m in merged_classes.values()
    ):
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
                    label=(None if is_pearson
                           else f"{PAIR_CLASS_LABEL['NN']}"),
                )
        # Responder × responder (RR) pairs: blue scatter + fit line + band.
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
        # All cells pooled (RR + NN + RN): one fit line over every pair.
        _fit_and_plot_subset(
            ax,
            dists, corrs,
            color=PLOT_PARAMS["corr_fit_color"],
            label_prefix="All cells pooled",
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
        )

    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    # The "[log1p distance axis]" title suffix is baked into the template via
    # {log1p_note}; the log1p switch only restyles the x-axis (label + ticks).
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])

    if fill.get("apply_log1p"):
        _apply_log1p_xaxis(ax)

    ax.legend(fontsize=P["legend_fontsize"], loc="lower right");


# =============================================================================
# Figure enumeration.
# =============================================================================
def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for every correlation-distance figure.

    For each (channel, method) and each pooled (method), yields BOTH the plain
    instance and its ``_log1p`` twin (same render fn, ``apply_log1p`` flag).
    """
    data = blob["data"]
    meta = blob["meta"]
    window_label = data["window_label"]
    exp_meta = meta["exp_name"]
    caveat_channel = meta["caveat_per_channel"]
    caveat_combined = meta["caveat_combined"]

    # ------------------------------------------------------------------ per-channel
    for entry in data["per_channel"]:
        if entry.get("status") != "ok":
            continue  # source drew an "insufficient data/samples" placeholder
        ch = entry["ch"]
        n_cells = entry["n_cells"]
        is_chamber = entry["is_chamber"]
        chamber = entry["chamber"]
        for method in ("pearson", "spearman"):
            spec_key = ("corr_vs_dist_channel_pearson" if method == "pearson"
                        else "corr_vs_dist_channel_spearman")
            clean = is_chamber and method == "pearson"
            mantel = entry["mantel"][method]
            base_payload = {
                "pw_dist": entry["pw_dist"],
                "pw_corr": entry["pw_corr_by_method"][method],
                "pair_classes": entry.get("pair_classes"),
                "clean_legend": clean,
                "mantel_all": mantel["all"],
                "mantel_rr": mantel["RR"],
            }
            # Resolve the clean/verbose title + ylabel HERE so figures_spec can
            # stay static templates ({title_main} / {ylabel_main}). Only the NRK
            # chamber Pearson panels take the caption-oriented "chamber {x}"
            # title + "Pearson r (...)" ylabel; everything else is labelled by
            # channel with the method in the title.
            if clean:
                title_main = (
                    f"Pairwise Pearson r vs. distance: chamber {chamber}"
                )
                ylabel_main = f"Pearson r ({window_label})"
            else:
                title_main = (
                    f"{ch}  ({n_cells} cells, {METHOD_LABEL[method]})"
                )
                ylabel_main = f"{ch} ({window_label})"
            base_fill = {
                "exp_name": exp_name,
                "ch": ch,
                "n_cells": n_cells,
                "chamber": chamber,
                "method_label": METHOD_LABEL[method],
                "window_label": window_label,
                "title_main": title_main,
                "ylabel_main": ylabel_main,
                "caveat": caveat_channel,
            }
            for log1p in (False, True):
                yield (
                    spec_key,
                    dict(base_payload),
                    dict(base_fill,
                         apply_log1p=log1p,
                         log1p_suffix=("_log1p" if log1p else ""),
                         log1p_note=("  [log1p distance axis]" if log1p else "")),
                )

    # ------------------------------------------------------------------ combined
    # Pool the per-channel vectors across channels (the source concatenated them
    # at render time; do it here so the render fn stays a single-axis draw).
    ok = [e for e in data["per_channel"] if e.get("status") == "ok"]
    if ok:
        for method in ("pearson", "spearman"):
            all_dist, all_corr = [], []
            all_classes = {"RR": [], "NN": [], "RN": []}
            for e in ok:
                all_dist.append(np.asarray(e["pw_dist"], dtype=float))
                all_corr.append(np.asarray(e["pw_corr_by_method"][method],
                                           dtype=float))
                pc = e.get("pair_classes")
                if pc is not None:
                    for tag in all_classes:
                        all_classes[tag].append(np.asarray(pc[tag]))
            dists = np.concatenate(all_dist) if all_dist else np.array([])
            corrs = np.concatenate(all_corr) if all_corr else np.array([])
            any_classes = any(len(v) > 0 for v in all_classes.values())
            merged_classes = (
                {tag: np.concatenate(v) for tag, v in all_classes.items()}
                if any_classes else None
            )
            cm = data["combined"][method]
            spec_key = ("corr_vs_dist_combined_pearson" if method == "pearson"
                        else "corr_vs_dist_combined_spearman")
            base_payload = {
                "pw_dist": dists,
                "pw_corr": corrs,
                "pair_classes": merged_classes,
                "method": method,
                "p_rr": cm["p_rr"],
                "p_all": cm["p_all"],
            }
            base_fill = {
                "exp_name": exp_name,
                "exp_name_disp": exp_meta,
                "method_label": METHOD_LABEL[method],
                "window_label": window_label,
                "caveat": caveat_combined,
            }
            for log1p in (False, True):
                yield (
                    spec_key,
                    dict(base_payload),
                    dict(base_fill,
                         apply_log1p=log1p,
                         log1p_suffix=("_log1p" if log1p else ""),
                         log1p_note=("  [log1p distance axis]" if log1p else "")),
                )
