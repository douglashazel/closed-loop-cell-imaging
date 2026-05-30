"""Render functions for the average-peak figures (reads average_peak.pkl).

For each DMSO experiment every per-stimulus response segment — the dF/F0 trace
from a stimulus onset to 10 min later (the inter-stimulus interval) — is pooled
across cells/channels, resampled onto a 0-10 min grid, and overlaid with the
mean ± 3 SEM:

  * ``average_peak``                  — all cells (faint per-segment overlay)
  * ``average_peak_responders``       — responder cells only (mean line only)
  * ``average_peak_responders_stim8`` — just the stim-#8 response window

Two cross-experiment companions overlay the per-experiment responder means
(PC3 vs C2C12, mean ± 1 SEM each) into the ``dmso_stim8_comparison`` bucket —
see :func:`build_combined`.

The analysis layer caches the RAW stacked segments; the low-coverage tail clip
(``MIN_COVERAGE_FRAC``) and the mean ± SEM are applied HERE via
:func:`_mean_sem_clipped`. Label/title/legend text lives in ``figures_spec.py``;
these functions pull it from the passed ``spec``.
"""
import os

import matplotlib.patheffects as pe
import numpy as np

from common.io_paths import load_analysis_cache
from plots._base import (
    FigureSpec,
    PLOT_PARAMS,
    clean_axes,
    legend_text,
    make_standalone,
    title_of,
    xlabel_of,
    ylabel_of,
)

NAME = "average_peak"

# Stimulus singled out for the stim-#8 derivative figures (mirrors the analysis
# layer; used only to fill the title/legend templates).
STIM8_INDEX = 8
# Frame timing means most segments stop just short of SEGMENT_MINUTES, so the
# extreme tail grid points are covered by only a handful of segments and their
# mean is unrepresentative (it dives toward baseline). Drop grid points whose
# segment coverage falls below this fraction of the best-covered point.
MIN_COVERAGE_FRAC = 0.5


def _mean_sem_clipped(stacked, grid):
    """Mean ± SEM over pooled segments, dropping low-coverage tail grid points.

    The extreme tail grid points are covered by only a handful of segments, so
    their mean dives unrepresentatively toward baseline (see
    ``MIN_COVERAGE_FRAC``). Returns ``(grid_kept, mean, sem)``.
    """
    counts = np.sum(~np.isnan(stacked), axis=0)
    keep = counts >= MIN_COVERAGE_FRAC * counts.max()
    grid = grid[keep]
    stacked = stacked[:, keep]
    counts = counts[keep]
    mean = np.nanmean(stacked, axis=0)
    sem = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(counts)
    return grid, mean, sem


def _draw_mean_band(ax, grid, mean_peak, sem, color, *, sem_mult,
                    mean_label=None, band_label=None):
    """Shade a mean ± ``sem_mult``·SEM band + the haloed mean line."""
    P = PLOT_PARAMS
    mean_lw = P["mean_lw"] * 1.7
    ax.fill_between(
        grid, mean_peak - sem_mult * sem, mean_peak + sem_mult * sem,
        color=color, alpha=0.25, linewidth=0, zorder=2, label=band_label,
    )
    mean_line, = ax.plot(
        grid, mean_peak, color=color, linewidth=mean_lw, zorder=3,
        label=mean_label,
    )
    # White halo so the average reads clearly over the dense overlay.
    mean_line.set_path_effects([
        pe.Stroke(linewidth=mean_lw + 2.4, foreground="white"),
        pe.Normal(),
    ])


def render_average_peak(ax, payload, spec, *, fill):
    """All-cells / responders-only average-peak overlay.

    ``payload`` carries the RAW stacked segments + grid; the coverage clip and
    mean ± 3 SEM are computed here. ``fill["show_cells"]`` draws the faint
    per-cell×stim segment overlay (all-cells figure); responders-only omits it.
    """
    P = PLOT_PARAMS
    clean_axes(ax)
    grid = payload["grid"]
    stacked = payload["stacked"]
    grid, mean_peak, sem = _mean_sem_clipped(stacked, grid)

    # Re-clip the raw rows to the kept grid for the faint per-segment overlay.
    if fill.get("show_cells"):
        counts = np.sum(~np.isnan(payload["stacked"]), axis=0)
        keep = counts >= MIN_COVERAGE_FRAC * counts.max()
        for row in payload["stacked"][:, keep]:
            ax.plot(grid, row, color=P["cell_color"], alpha=0.05,
                    linewidth=P["cell_lw"], zorder=1)

    _draw_mean_band(
        ax, grid, mean_peak, sem, P["pooled_mean_color"], sem_mult=3,
        mean_label=legend_text(spec, "mean", fill),
        band_label=legend_text(spec, "band", fill),
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def render_average_peak_stim8(ax, payload, spec, *, fill):
    """Stim-#8 responder average-peak overlay (mean ± 3 SEM, line only)."""
    P = PLOT_PARAMS
    clean_axes(ax)
    grid = payload["grid"]
    grid, mean_peak, sem = _mean_sem_clipped(payload["stacked"], grid)
    _draw_mean_band(
        ax, grid, mean_peak, sem, P["pooled_mean_color"], sem_mult=3,
        mean_label=legend_text(spec, "mean", fill),
        band_label=legend_text(spec, "band", fill),
    )
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def render_average_peak_combined(ax, payload, spec, *, fill):
    """Overlay every DMSO experiment's responder mean ± 1 SEM on one axis.

    ``payload`` is a list of ``(exp_name, stacked, grid)`` triples. Each
    experiment gets a ±1 SEM band (lighter than the per-experiment ±3 SEM so
    two bands stay legible) coloured by ``PLOT_PARAMS["colors"]``. The legend
    label template is read from ``spec`` with each experiment's name/count.
    """
    P = PLOT_PARAMS
    clean_axes(ax)
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
    mean_lw = P["mean_lw"] * 1.7
    for i, (exp_name, stacked, grid) in enumerate(payload):
        color = P["colors"][i % len(P["colors"])]
        g, mean_peak, sem = _mean_sem_clipped(stacked, grid)
        ax.fill_between(g, mean_peak - sem, mean_peak + sem,
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        line, = ax.plot(
            g, mean_peak, color=color, linewidth=mean_lw, zorder=3,
            label=legend_text(
                spec, "exp",
                {**fill, "exp_name": exp_name, "n": int(stacked.shape[0])},
            ),
        )
        # White halo so each mean reads clearly where the bands overlap.
        line.set_path_effects([
            pe.Stroke(linewidth=mean_lw + 2.4, foreground="white"),
            pe.Normal(),
        ])
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for each per-exp average-peak figure."""
    data = blob["data"]
    meta = blob["meta"]
    cell_line = meta["cell_line"]
    n_channels = data["n_channels"]
    grid = data["grid"]

    # All cells.
    all_stacked = data["all_segments_stacked"]
    yield (
        "average_peak",
        {"grid": grid, "stacked": all_stacked},
        {"exp_name": exp_name, "n_channels": n_channels,
         "n_seg": int(all_stacked.shape[0]), "show_cells": True},
    )

    # Responders only.
    resp_stacked = data["resp_segments_stacked"]
    if resp_stacked is not None:
        yield (
            "average_peak_responders",
            {"grid": grid, "stacked": resp_stacked},
            {"exp_name": exp_name, "n_channels": n_channels,
             "n_seg": int(resp_stacked.shape[0]), "show_cells": False},
        )

    # Stim-#8 responders.
    stim8_stacked = data["stim8_stacked"]
    if stim8_stacked is not None:
        yield (
            "average_peak_responders_stim8",
            {"grid": grid, "stacked": stim8_stacked},
            {"exp_name": exp_name, "cell_line": cell_line,
             "n_channels": n_channels,
             "n_seg": int(stim8_stacked.shape[0])},
        )


# =============================================================================
# Cross-experiment combined figures (PC3 vs C2C12).
# =============================================================================
COMBINED_BUCKET = "dmso_stim8_comparison"


def _combined_spec(key):
    """Build the FigureSpec for a combined figure inline (kept here so the
    cross-experiment label text stays in FigureSpec form, like per-exp figs)."""
    if key == "average_peak_responders_combined":
        return FigureSpec(
            id="average_peak_responders_combined",
            analysis="average_peak", scope="cross_experiment",
            render=render_average_peak_combined,
            title=(
                "DMSO average response peak — PC3 vs C2C12 "
                "(responders only, all stims)\nmean ± 1 SEM per experiment"
            ),
            xlabel="Time since stimulus onset (min)", ylabel="dF/F₀",
            legend={"exp": "{exp_name}  (n={n} responder cell×stim segments)"},
        )
    return FigureSpec(
        id="average_peak_responders_stim8_combined",
        analysis="average_peak", scope="cross_experiment",
        render=render_average_peak_combined,
        title=(
            f"DMSO stimulus #{STIM8_INDEX} average response peak — "
            "PC3 vs C2C12 (responders only)\nmean ± 1 SEM per experiment"
        ),
        xlabel="Time since stimulus onset (min)", ylabel="dF/F₀",
        legend={"exp": "{exp_name}  (n={n} responder cell segments)"},
    )


def build_combined(experiments):
    """Render the two PC3-vs-C2C12 combined average-peak figures.

    Loads each DMSO experiment's ``average_peak`` cache, overlays the per-
    experiment responder mean ± 1 SEM, and writes both figures into the
    ``dmso_stim8_comparison`` output bucket (``fill["exp_name"]`` is the bucket
    dir). Only renders when ≥2 DMSO caches are present (mirrors the source's
    ``len >= 2`` guard); silently defers otherwise.

    ``experiments`` may be a list of names or a name→cfg dict; this only needs
    the names, since the DMSO/responder filtering already happened at analysis
    time (non-DMSO experiments never wrote an ``average_peak`` cache).
    """
    resp_per_exp = []   # (exp_name, resp_stacked, grid)
    stim8_per_exp = []  # (exp_name, stim8_stacked, grid)
    for name in experiments:
        try:
            blob = load_analysis_cache(name, "average_peak")
        except FileNotFoundError:
            continue  # not a DMSO experiment / not yet analyzed
        data = blob["data"]
        grid = data["grid"]
        if data.get("resp_segments_stacked") is not None:
            resp_per_exp.append((name, data["resp_segments_stacked"], grid))
        if data.get("stim8_stacked") is not None:
            stim8_per_exp.append((name, data["stim8_stacked"], grid))

    if len(resp_per_exp) >= 2:
        spec = _combined_spec("average_peak_responders_combined")
        make_standalone(spec, resp_per_exp, {"exp_name": COMBINED_BUCKET})
        print(
            f"  average peak responders (combined): {len(resp_per_exp)} "
            f"experiment(s) overlaid → {COMBINED_BUCKET}/"
        )
    else:
        print(
            f"  responders combined overlay: only {len(resp_per_exp)} DMSO "
            "cache(s) present — deferring."
        )

    if len(stim8_per_exp) >= 2:
        spec = _combined_spec("average_peak_responders_stim8_combined")
        make_standalone(spec, stim8_per_exp, {"exp_name": COMBINED_BUCKET})
        print(
            f"  stim-#{STIM8_INDEX} average peak (combined): "
            f"{len(stim8_per_exp)} experiment(s) overlaid → {COMBINED_BUCKET}/"
        )
    else:
        print(
            f"  stim-#{STIM8_INDEX} combined overlay: only "
            f"{len(stim8_per_exp)} DMSO cache(s) present — deferring."
        )
