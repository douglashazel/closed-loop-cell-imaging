"""Render functions for the response-violin figures (reads analysis_cache/<exp>/response_violins.pkl).

Three standalone single-axis figures are produced per metric in {height, width}:
  * ``pooled_response_violin_{metric}_dff``            — asymmetric half-violin
    + notched boxplot + jittered (grey) scatter + mean markers,
  * ``pooled_response_violin_{metric}_dff_responders`` — same violin with each
    responder cell's scatter point highlighted,
  * ``pooled_response_violin_{metric}_dff_train_means`` — per-replicate
    train-mean lines with a first→last significance bracket.

``_draw_half_violin_with_box`` and ``_plot_replicate_train_means`` are ported
verbatim from the original ``response_violins.py``, retargeted from a
self-created figure onto the passed-in ``ax`` (the driver owns the figure and
saves it). Label/title/legend text lives centrally in ``figures_spec.py``; these
functions pull it from the passed ``spec``.

The source rendered ``stats_text`` and the inferential ``caveat`` as
commented-out elements (upper-left box / figure footnote); they are cached but
NOT drawn here, matching the source's visible output. The caveat is wired into
each FigureSpec.caveat so re-enabling it is a one-line change.
"""
import numpy as np

from plots._base import (
    PLOT_PARAMS,
    clean_axes,
    legend_text,
    title_of,
    xlabel_of,
    ylabel_of,
)

NAME = "response_violins"

# (filename suffix, responder-highlight flag) for the two violin variants.
VIOLIN_SUBSETS = [("", False), ("_responders", True)]

# Asymmetric-violin geometry — verbatim from the source module constants.
_VIOLIN_BOX_OFFSET = 0.18
_VIOLIN_BOX_WIDTH = 0.18
_VIOLIN_SCATTER_JITTER = 0.045


def _sig_stars(p):
    """Significance label for a p-value: ``***`` / ``**`` / ``*`` / ``ns``.

    Returns ``"n/a"`` when ``p`` is missing or non-finite (e.g. only one
    biological replicate, so no replicate-level test was run).
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def render_violin(ax, payload, spec, *, fill):
    """Asymmetric half-violin + notched box + jittered/responder scatter + means.

    ``payload`` supplies ``violin_data`` (ragged list of 1-D arrays), the
    parallel ``responder_data``, and the per-stim ``x_labels``. When
    ``payload["highlight_responders"]`` is True, responder cells' scatter points
    are drawn in the responder color on top of the greyed non-responder cloud.
    Ported verbatim from ``_draw_half_violin_with_box`` onto the passed ``ax``.
    """
    P = PLOT_PARAMS
    clean_axes(ax)

    violin_data = payload["violin_data"]
    x_label = payload["x_labels"]
    responder_data = (
        payload["responder_data"] if payload.get("highlight_responders") else None
    )
    n_cat = len(violin_data)
    non_empty_idx = [i for i, v in enumerate(violin_data) if len(v) > 0]
    if not non_empty_idx:
        return

    box_x = [i - _VIOLIN_BOX_OFFSET for i in non_empty_idx]
    box_data = [np.asarray(violin_data[i]) for i in non_empty_idx]

    rng = np.random.default_rng(42)
    resp_label_used = False
    for col_idx, bx in zip(non_empty_idx, box_x):
        vd = np.asarray(violin_data[col_idx])
        xs = bx + rng.uniform(
            -_VIOLIN_SCATTER_JITTER, _VIOLIN_SCATTER_JITTER, size=len(vd)
        )
        if responder_data is not None:
            rmask = np.asarray(responder_data[col_idx], dtype=bool)
        else:
            rmask = np.zeros(len(vd), dtype=bool)
        ax.scatter(
            xs[~rmask], vd[~rmask],
            color=P["scatter_color"],
            alpha=P["scatter_alpha"],
            s=P["scatter_size"],
            zorder=2, linewidths=0, rasterized=True,
        )
        if rmask.any():
            ax.scatter(
                xs[rmask], vd[rmask],
                color=P["responder_color"],
                edgecolors=P["responder_edge"],
                alpha=0.9,
                s=P["scatter_size"] * 1.7,
                linewidths=0.5, zorder=5, rasterized=True,
                label=None if resp_label_used else legend_text(spec, "responder", fill),
            )
            resp_label_used = True

    bp = ax.boxplot(
        box_data,
        positions=box_x,
        widths=_VIOLIN_BOX_WIDTH,
        notch=True,
        bootstrap=None,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color=P["median_color"], linewidth=2.0
        ),
        boxprops=dict(
            facecolor=P["violin_face"],
            edgecolor=P["violin_edge"],
            linewidth=1.2,
        ),
        whiskerprops=dict(color=P["violin_edge"], linewidth=1.0),
        capprops=dict(color=P["violin_edge"], linewidth=1.0),
        zorder=3,
    )
    median_handles = bp.get("medians", [])
    for i, handle in enumerate(median_handles):
        handle.set_label(legend_text(spec, "median", fill) if i == 0 else None)

    vp = ax.violinplot(
        box_data,
        positions=non_empty_idx,
        showmedians=False,
        showextrema=False,
        showmeans=False,
    )
    for body, pos in zip(vp["bodies"], non_empty_idx):
        for path in body.get_paths():
            verts = path.vertices
            verts[verts[:, 0] < pos, 0] = pos
        body.set_facecolor(P["violin_face"])
        body.set_edgecolor(P["violin_edge"])
        body.set_alpha(0.85)
        body.set_zorder(3)

    means = [float(np.mean(v)) for v in box_data]
    ax.scatter(
        non_empty_idx, means,
        marker="_",
        color=P["mean_marker_color"],
        s=200, linewidths=2.5,
        zorder=6, label=legend_text(spec, "mean", fill),
    )

    ax.legend(fontsize=P["legend_fontsize_large"], loc="center right");

    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])
    ax.set_xticks(range(n_cat))
    ax.set_xticklabels([str(lbl) for lbl in x_label], fontsize=9)
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])


def render_train_means(ax, payload, spec, *, fill):
    """Per-replicate train-mean lines + first→last significance bracket.

    ``payload["chan_train_means"]`` has shape ``(n_channels, n_trains)`` — one
    green line per biological replicate across the stimulus trains, ramped
    dark→light. ``payload["train_p"]`` is the replicate-level one-sample t-test
    p-value for the first→last train change, annotated as a bracket spanning
    train 1 to train N. Ported verbatim from ``_plot_replicate_train_means``.
    """
    P = PLOT_PARAMS
    clean_axes(ax)

    chan_means = np.asarray(payload["chan_train_means"], dtype=float)
    train_p = payload["train_p"]
    n_ch, n_tr = chan_means.shape
    train_x = np.arange(n_tr)
    greens = P["replicate_greens"]
    for ci in range(n_ch):
        ax.plot(
            train_x, chan_means[ci],
            color=greens[ci % len(greens)],
            linewidth=2.4, marker="D", markersize=9,
            markeredgecolor="#222222", markeredgewidth=0.7,
            label=legend_text(spec, "rep", dict(fill, rep_num=ci + 1)),
        )
    ax.set_xticks(train_x)
    ax.set_xticklabels([f"Train {t + 1}" for t in range(n_tr)])
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])

    # Significance bracket for the first→last train change (replicate level).
    if n_tr >= 2:
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) or 1.0
        bar_y = y1 + span * 0.08
        ax.plot(
            [train_x[0], train_x[-1]], [bar_y, bar_y],
            color="#222222", linewidth=1.4,
        )
        ax.text(
            (train_x[0] + train_x[-1]) / 2.0, bar_y + span * 0.03,
            _sig_stars(train_p),
            ha="center", va="bottom", fontsize=14,
            fontweight=P["title_fontweight"],
        )
        ax.set_ylim(y0, bar_y + span * 0.22)
    ax.legend(fontsize=P["legend_fontsize"], loc="best");


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for every response-violin figure."""
    data = blob["data"]

    for metric, bundle in data.items():
        base_fill = {
            "exp_name": exp_name,
            "metric": metric,
            "y_label": bundle["y_label"],
            "title_core": bundle["title_core"],
            "width_cap_note": bundle["width_cap_note"],
            "n_total_cells": bundle["n_total_cells"],
            "n_channels": bundle["n_channels"],
            "n_total_responders": bundle["n_total_responders"],
        }

        # The two violin variants (all cells / responders highlighted).
        for suffix, highlight in VIOLIN_SUBSETS:
            payload = {
                "violin_data": bundle["violin_data"],
                "responder_data": bundle["responder_data"],
                "x_labels": bundle["x_labels"],
                "highlight_responders": highlight,
            }
            spec_key = ("response_violin_responders" if highlight
                        else "response_violin")
            yield (spec_key, payload, dict(base_fill))

        # Per-replicate train means — only when a replicate-level structure
        # exists (>= 2 channels and fixed trains; matches the source guard).
        ctm = bundle["chan_train_means"]
        if ctm is not None and np.asarray(ctm).size:
            yield (
                "response_violin_train_means",
                {"chan_train_means": ctm, "train_p": bundle["train_p"]},
                dict(base_fill),
            )
