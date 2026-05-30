"""Nature-style multi-panel mosaics assembled from the standalone figures.

A mosaic reuses the SAME ``render_*`` functions the standalone figures use,
drawing each onto a ``subplot_mosaic`` axes (or a ``SubFigure`` for the
multi-panel ``responder_diagnostic`` exception). Because every render function
sets its own title/labels at the locked rcParams point sizes, the panels keep
identical fontsizes whether standalone or composed here.

Add a mosaic by appending to ``MOSAICS``: give a ``subplot_mosaic`` ``layout``,
a ``figsize``, and a ``cells`` map from each mosaic key to
``(spec_key, instance_match)`` — where ``instance_match`` selects ONE figure
instance from that spec's ``iter_figures`` output (e.g. a specific channel or
subset). Build with ``make_figures.py --mosaics <name>``.

A cell tuple may carry an optional THIRD element, ``(spec_key, instance_match,
exp_name)``, naming the experiment that cell is pulled from (it overrides the
experiment ``build_mosaic`` is invoked for). That is what lets a single mosaic
mix panels from several experiments — pair it with ``experiments`` (below) so
the cross-experiment mosaic is assembled exactly once.

Optional per-mosaic keys (all default to the prior behaviour, so existing
mosaics are unchanged):

* ``gridspec_kw``  — forwarded to ``subplot_mosaic`` (e.g. unequal panel
  heights via ``{"height_ratios": [...]}``).
* ``titles``       — ``{mosaic_key: str}`` overriding the title the render
  function drew for that cell (drawn at the locked title point size/weight).
* ``share_xlim`` / ``share_ylim`` — ``True`` locks every cell to the UNION of
  the cells' autoscaled x-/y-ranges (one shared scale, nothing clipped); a list
  of mosaic keys (e.g. ``["a", "b", "c"]``) shares only those cells.
* ``xlims`` / ``ylims`` — ``{mosaic_key: (lo, hi)}`` forcing an explicit range
  on individual cells, applied after (and overriding) any shared value.
* ``legend_loc`` — ``{mosaic_key: loc}`` relocating that cell's legend (e.g.
  ``"upper left"``) without touching the shared render function's default.
* ``hide_legend`` — ``True`` (every cell) or a list of mosaic keys whose legend
  the render function drew is removed entirely.
* ``hide_ticks`` — ``True`` (every cell) or a list of mosaic keys to strip of
  both x- and y-axis tick marks AND tick labels (the axis labels stay).
* ``panel_labels`` — set ``False`` to suppress the a/b/c… panel letters (e.g.
  when each panel's title already names it).
* ``panel_label_x`` — ``{mosaic_key: x}`` overriding the axes-fraction x of that
  cell's panel letter (default ``PLOT_PARAMS["panel_label_x"]`` = -0.12; a less
  negative value nudges the letter rightward, e.g. for right-column panels).
* ``xlabels`` / ``ylabels`` — ``{mosaic_key: str}`` overriding the axis label
  the render function drew for that cell.
* ``axis_label_fontsize`` — point size for the ``xlabels``/``ylabels`` overrides
  (default ``PLOT_PARAMS["axis_label_fontsize"]`` = 8).
* ``box_aspect`` — ``{mosaic_key: float}`` forcing the axes box height/width
  ratio (1.0 = square; >1 = taller & narrower, visually squishing the x-axis).
* ``xtick_fontsize`` — ``{mosaic_key: pt}`` resizing that cell's x-tick labels
  (e.g. to match a render function that hard-codes its own tick size).
* ``experiments`` — list of experiment names this mosaic may build for; for any
  other experiment ``build_mosaic`` raises ``KeyError`` so the driver skips it
  cleanly. Needed when the source panels exist for several experiments (so the
  KeyError-on-missing-channel trick won't restrict it) but every mosaic shares
  one filename-per-name folder, so an unrestricted build would let a later
  experiment overwrite an earlier one's PNG.

Every mosaic PNG is written to the shared ``<OUT_ROOT>/mosaics/`` folder.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_paths import fig_path, load_analysis_cache, save_fig  # noqa: E402
from figures_spec import FIGURES  # noqa: E402
from style import PLOT_PARAMS, add_panel_label, apply_style  # noqa: E402

from plots import (  # noqa: E402
    average_peak, clustering, correlation_distance, dff, learning_scores,
    nrk_hardware_log, response_violins, responder_diagnostic,
)

_MODULES_BY_NAME = {
    m.NAME: m for m in [
        dff, clustering, average_peak, correlation_distance, response_violins,
        learning_scores, nrk_hardware_log, responder_diagnostic,
    ]
}


# =============================================================================
# Mosaic definitions.  cells: {mosaic_key: (spec_key, instance_match)}.
# =============================================================================
MOSAICS = {
    # Reassembles the old 2-row <ch>_dff figure (corrected on top, dF/F0 below)
    # from the two decomposed standalone panels — also handy for verifying the
    # decomposition matches the legacy combined PNG.
    "dff_stack_ch1": {
        "layout": [["a"], ["b"]],
        "figsize": (PLOT_PARAMS["width_full"], 8.0),
        "suptitle": "{exp_name} / channel 1 — corrected + dF/F₀",
        "cells": {
            "a": ("dff_raw", {"ch": "channel 1", "subset_suffix": ""}),
            "b": ("dff_norm", {"ch": "channel 1", "subset_suffix": ""}),
        },
    },
    # Example overview: PCA + UMAP side by side with the responder-pooled mean.
    "clustering_overview": {
        "layout": [["a", "b"], ["c", "c"]],
        "figsize": (PLOT_PARAMS["width_full"], 5.6),
        "suptitle": "{exp_name} — embedding + responder mean",
        "cells": {
            "a": ("pooled_pca_only", {}),
            "b": ("pooled_umap_only", {}),
            "c": ("dff_mean_pooled_responders", {}),
        },
    },
    # c2c12 chamber stack: the three per-channel dF/F0 trace panels (relabelled
    # Chamber A/B/C) over the pooled responder mean — one shared x- and y-scale
    # so the panels line up, the pooled panel a touch taller. c2c12-only (other
    # experiments lack channels 2/3, so build_mosaic skips them via KeyError).
    "c2c12_chambers_dff_stack": {
        "layout": [["a"], ["b"], ["c"], ["d"]],
        "figsize": (PLOT_PARAMS["width_full"], 9.0),
        # Pooled-responder panel (d) a touch taller; chamber traces shorter.
        "gridspec_kw": {"height_ratios": [1.0, 1.0, 1.0, 1.4]},
        # The chamber titles label panels a–c, so skip the a/b/c… letters.
        "panel_labels": False,
        "titles": {"a": "Chamber A", "b": "Chamber B", "c": "Chamber C"},
        # Move the three chamber legends to the upper left (default upper right).
        "legend_loc": {"a": "upper left", "b": "upper left", "c": "upper left"},
        # Shared x-range across all four; the three chambers share one y-range,
        # while the pooled responder mean gets its own zoomed-in y-range.
        "share_xlim": True,
        "share_ylim": ["a", "b", "c"],
        "ylims": {"d": (-0.1, 0.5)},
        "cells": {
            "a": ("dff_norm", {"ch": "channel 1", "subset_suffix": ""}),
            "b": ("dff_norm", {"ch": "channel 2", "subset_suffix": ""}),
            "c": ("dff_norm", {"ch": "channel 3", "subset_suffix": ""}),
            "d": ("dff_mean_pooled_responders", {}),
        },
    },
    # c2c12 2×2 overview: correlation-vs-distance + PCA on the top row, the
    # single-cell response violins + per-replicate train means on the bottom.
    # c2c12-only (the other experiments lack one of these panels, so they skip).
    "c2c12_corr_pca_responses": {
        "layout": [["a", "b"], ["c", "d"]],
        # Wider than the 6.5 in page width (height unchanged) so the wide a/c
        # panels aren't squished width-wise.
        "figsize": (9.5, 6.5),
        # Give the wide a/c panels a fatter left column so they extend toward
        # the (aspect-constrained) b/d panels in the narrower right column.
        "gridspec_kw": {"width_ratios": [1.6, 1.0]},
        # b (PCA) and d (train means) are both square, so they take the same
        # width in the right column (and d's title gets room from its label).
        "box_aspect": {"b": 1.0, "d": 1.0},
        "titles": {
            "c": "Single cell responses to each stimulus",
            "d": "Mean response vs. stimulus train",
        },
        "ylabels": {"c": "Response peak height (max - baseline)"},
        "legend_loc": {"c": "upper left"},
        # render_violin hard-codes 9 pt x-ticks; drop c back to the locked tick
        # size so it matches panel a (and the rest of the house style).
        "xtick_fontsize": {"c": PLOT_PARAMS["tick_fontsize"]},
        "cells": {
            "a": ("corr_vs_dist_combined_pearson", {"log1p_suffix": ""}),
            "b": ("pooled_pca_only", {}),
            "c": ("response_violin", {"metric": "height"}),
            "d": ("response_violin_train_means", {"metric": "height"}),
        },
    },
    # c2c12 learning-score panel: one row per learning measure, each pairing the
    # observed-vs-shuffled score histogram (left) with its permutation test
    # (right) — habituation, sensitization, then anticipation trains 1 and 2 (all
    # on the height metric). The learning figures exist for both DMSO
    # experiments, so restrict to c2c12 via ``experiments`` (mosaics share one
    # filename folder, so an unrestricted build would let pc3 overwrite this).
    "c2c12_learning_scores": {
        "layout": [
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["g", "h"],
        ],
        "figsize": (PLOT_PARAMS["width_full"], 9.5),
        "experiments": ["c2c12_dmso_09APR26"],
        # Both anticipation permtests render the bare title "Anticipation
        # permutation test"; name the train so the two rows are distinguishable.
        "titles": {
            "f": "Anticipation permutation test: train 1",
            "h": "Anticipation permutation test: train 2",
        },
        "cells": {
            "a": ("learning_score_hist",
                  {"measure_key": "habituation", "metric": "height"}),
            "b": ("learning_score_permtest",
                  {"measure_key": "habituation", "metric": "height"}),
            "c": ("learning_score_hist",
                  {"measure_key": "sensitization", "metric": "height"}),
            "d": ("learning_score_permtest",
                  {"measure_key": "sensitization", "metric": "height"}),
            "e": ("learning_anticipation_hist", {"train_idx": 1}),
            "f": ("learning_anticipation_permtest", {"train_idx": 1}),
            "g": ("learning_anticipation_hist", {"train_idx": 2}),
            "h": ("learning_anticipation_permtest", {"train_idx": 2}),
        },
    },
    # Cross-experiment responder overview: one DMSO cell line per row — the
    # pooled responder-mean trace (left) beside its mean response to stimulus #8
    # (right), c2c12 on top, pc3 below. Every cell names its own source
    # experiment (3rd tuple element), so this is assembled from both caches at
    # once; ``experiments`` anchors it to a single build pass (it reads each
    # cell's cache straight from disk, independent of the run's experiment list).
    # Each panel title already carries its cell line. All four y-axes are
    # relabelled "fluorescence".
    "dmso_responder_overview": {
        "layout": [
            ["a", "b"],
            ["c", "d"],
        ],
        "figsize": (PLOT_PARAMS["width_full"], 5.0),
        "experiments": ["c2c12_dmso_09APR26"],
        # Nudge the right-column letters rightward (default -0.12 sits them out
        # in the inter-column gap, away from their panels).
        "panel_label_x": {"b": -0.05, "d": -0.05},
        "ylabels": {
            "a": "fluorescence", "b": "fluorescence",
            "c": "fluorescence", "d": "fluorescence",
        },
        "cells": {
            "a": ("dff_mean_pooled_responders", {}, "c2c12_dmso_09APR26"),
            "b": ("average_peak_responders_stim8", {}, "c2c12_dmso_09APR26"),
            "c": ("dff_mean_pooled_responders", {}, "pc3_dmso_23MAR26"),
            "d": ("average_peak_responders_stim8", {}, "pc3_dmso_23MAR26"),
        },
    },
    # c2c12 channel-3 dF/F₀ pair: the corrected-fluorescence trace stack (left)
    # beside its normalized dF/F₀ stack (right), stripped to bare traces for a
    # schematic-style row — no title, no legend, no panel letters, no ticks/tick
    # labels, x clipped to the first 90 min. c2c12-only (channel 3 exists for no
    # other experiment, so build_mosaic skips them via KeyError; "experiments"
    # also anchors the shared-folder build).
    "c2c12_ch3_dff_pair": {
        # A spacer column ("." = empty cell) opens a wider gap between the two
        # panels; the middle width_ratio sizes it relative to a panel (1.0).
        "layout": [["a", ".", "b"]],
        "figsize": (PLOT_PARAMS["width_full"], 3.0),
        "gridspec_kw": {"width_ratios": [1.0, 0.25, 1.0]},
        "experiments": ["c2c12_dmso_09APR26"],
        "panel_labels": False,
        "hide_legend": True,
        "hide_ticks": True,
        # Drop the titles the render fns draw (set_title("") clears them).
        "titles": {"a": "", "b": ""},
        "xlabels": {"a": "time", "b": "time"},
        "ylabels": {"a": "fluorescence", "b": "normalized fluorescence"},
        # Bigger axis labels than the 8 pt house default for this bare panel.
        "axis_label_fontsize": 12,
        # Clip both panels' x-axis to the first 90 min.
        "xlims": {"a": (0, 90), "b": (0, 90)},
        "cells": {
            "a": ("dff_raw", {"ch": "channel 3", "subset_suffix": ""}),
            "b": ("dff_norm", {"ch": "channel 3", "subset_suffix": ""}),
        },
    },
    # NRK hardware-feedback luminosity logs as a 2×2 chamber grid: channel 1
    # (chambers A, C) on top, channel 2 (chambers B, D) below. Each panel's title
    # names its chamber. NRK-only — the other experiments lack the
    # nrk_hardware_log cache, so build_mosaic skips them via FileNotFoundError.
    "nrk_chambers_hw_log": {
        # A spacer column ("." = empty cell) between the two figure columns opens
        # a wider gap and shifts b/d rightward; the middle width_ratio sets its
        # size relative to a panel (1.0).
        "layout": [
            ["a", ".", "b"],
            ["c", ".", "d"],
        ],
        "figsize": (PLOT_PARAMS["width_full"], 4.0),
        "gridspec_kw": {"width_ratios": [1.0, 0.25, 1.0]},
        # No shared y-scale: each chamber keeps the per-channel y-range the render
        # fn sets from payload["y_lim"] (e.g. A ~52.2-60.4, C ~55.6-64.1), so every
        # panel frames its own trace + setpoint rather than sharing one scale.
        # Right-column legends (b, d) sit in the bottom-right corner.
        "legend_loc": {"b": "lower right", "d": "lower right"},
        "cells": {
            "a": ("nrk_hardware_log", {"ch": "channel 1 A"}),
            "b": ("nrk_hardware_log", {"ch": "channel 1 C"}),
            "c": ("nrk_hardware_log", {"ch": "channel 2 B"}),
            "d": ("nrk_hardware_log", {"ch": "channel 2 D"}),
        },
    },
    # NRK per-chamber dF/F₀ + correlation-vs-distance, one chamber per row: the
    # chamber's dF/F₀ trace stack (left) beside its pairwise Pearson-r vs distance
    # scatter (right). Rows follow the hardware-log grid order — channel 1
    # chambers A, C then channel 2 chambers B, D. NO shared y-scale: every panel
    # autoscales its own y-axis (the dF/F₀ traces and the r-vs-distance clouds
    # live on unrelated scales), so share_ylim is intentionally left unset. The
    # "Inferential unit: cell pair…" footnote the standalone corr panels print is
    # drawn by make_standalone, never by the render fn, so it doesn't appear here.
    # NRK-only — the chamber channel names ("channel 1 A", …) exist for no other
    # experiment, so build_mosaic skips the rest via KeyError; "experiments"
    # anchors it so a shared-folder build can't overwrite it.
    "nrk_chambers_dff_corr": {
        "layout": [
            ["a", "b"],
            ["c", "d"],
            ["e", "f"],
            ["g", "h"],
        ],
        "figsize": (PLOT_PARAMS["width_full"], 9.0),
        "experiments": ["nrk_acid_13APR26"],
        # Left column (dF/F₀ traces) legends in the bottom right; right column
        # (r-vs-distance scatters) legends in the bottom left.
        "legend_loc": {
            "a": "lower right", "c": "lower right",
            "e": "lower right", "g": "lower right",
            "b": "lower left", "d": "lower left",
            "f": "lower left", "h": "lower left",
        },
        "cells": {
            "a": ("dff_norm", {"ch": "channel 1 A", "subset_suffix": ""}),
            "b": ("corr_vs_dist_channel_pearson",
                  {"ch": "channel 1 A", "log1p_suffix": ""}),
            "c": ("dff_norm", {"ch": "channel 1 C", "subset_suffix": ""}),
            "d": ("corr_vs_dist_channel_pearson",
                  {"ch": "channel 1 C", "log1p_suffix": ""}),
            "e": ("dff_norm", {"ch": "channel 2 B", "subset_suffix": ""}),
            "f": ("corr_vs_dist_channel_pearson",
                  {"ch": "channel 2 B", "log1p_suffix": ""}),
            "g": ("dff_norm", {"ch": "channel 2 D", "subset_suffix": ""}),
            "h": ("corr_vs_dist_channel_pearson",
                  {"ch": "channel 2 D", "log1p_suffix": ""}),
        },
    },
}


def _find_instance(exp_name, spec_key, match):
    """Return ``(spec, payload, fill)`` for the single figure instance of
    ``spec_key`` whose ``fill`` matches every key/value in ``match``."""
    spec = FIGURES[spec_key]
    mod = _MODULES_BY_NAME[spec.analysis]
    blob = load_analysis_cache(exp_name, spec.analysis)
    for key, payload, fill in mod.iter_figures(blob, exp_name):
        if key != spec_key:
            continue
        if all(fill.get(k) == v for k, v in match.items()):
            return spec, payload, fill
    raise KeyError(
        f"no instance of '{spec_key}' matching {match} for {exp_name}"
    )


def _share_axis_limits(axd, m):
    """Lock cells to a common x-/y-range, then apply per-cell limit overrides.

    ``share_xlim`` / ``share_ylim`` select which cells share a scale: ``True``
    means every cell, a list of mosaic keys means only those. The shared range
    is the UNION of those cells' autoscaled limits (read after rendering) so no
    panel's data is clipped. ``xlims`` / ``ylims`` (``{key: (lo, hi)}``) then
    force explicit limits on individual cells, overriding any shared value. All
    keys default off/empty, so mosaics that don't set them are untouched.
    """
    def _members(flag):
        if flag is True:
            return list(m["cells"])
        return list(flag) if flag else []

    for share, get_lim, set_lim in (
        (m.get("share_xlim"), "get_xlim", "set_xlim"),
        (m.get("share_ylim"), "get_ylim", "set_ylim"),
    ):
        axes = [axd[k] for k in _members(share)]
        if not axes:
            continue
        lo = min(getattr(ax, get_lim)()[0] for ax in axes)
        hi = max(getattr(ax, get_lim)()[1] for ax in axes)
        for ax in axes:
            getattr(ax, set_lim)(lo, hi)

    for mkey, lim in m.get("xlims", {}).items():
        axd[mkey].set_xlim(*lim)
    for mkey, lim in m.get("ylims", {}).items():
        axd[mkey].set_ylim(*lim)


def build_mosaic(name, exp_name):
    """Assemble mosaic ``name`` for one experiment and save the PNG."""
    if name not in MOSAICS:
        raise KeyError(f"unknown mosaic '{name}'. Known: {list(MOSAICS)}")
    m = MOSAICS[name]
    allowed = m.get("experiments")
    if allowed is not None and exp_name not in allowed:
        # Skipped cleanly by the driver (KeyError) — keeps one-experiment mosaics
        # from overwriting each other in the shared mosaics/ folder.
        raise KeyError(
            f"mosaic '{name}' is restricted to {allowed}; skipping '{exp_name}'"
        )
    apply_style()
    fig, axd = plt.subplot_mosaic(
        m["layout"], figsize=m["figsize"], layout="constrained",
        gridspec_kw=m.get("gridspec_kw"),
    )
    titles = m.get("titles", {})
    for mkey, cell in m["cells"].items():
        # Cells are (spec_key, match) or (spec_key, match, exp_name); the
        # optional 3rd element pulls that panel from another experiment so one
        # mosaic can mix experiments (defaults to the build's exp_name).
        spec_key, match = cell[0], cell[1]
        cell_exp = cell[2] if len(cell) > 2 else exp_name
        spec, payload, fill = _find_instance(cell_exp, spec_key, match)
        if spec.multi_panel:
            raise ValueError(
                f"mosaic cell '{mkey}' references multi-panel figure "
                f"'{spec_key}'; compose those via fig.subfigures instead."
            )
        spec.render(axd[mkey], payload, spec, fill=fill)
        if mkey in titles:
            axd[mkey].set_title(
                titles[mkey], fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
        label_fs = m.get("axis_label_fontsize", PLOT_PARAMS["axis_label_fontsize"])
        if mkey in m.get("xlabels", {}):
            axd[mkey].set_xlabel(m["xlabels"][mkey], fontsize=label_fs)
        if mkey in m.get("ylabels", {}):
            axd[mkey].set_ylabel(m["ylabels"][mkey], fontsize=label_fs)
        if mkey in m.get("box_aspect", {}):
            axd[mkey].set_box_aspect(m["box_aspect"][mkey])
        if mkey in m.get("xtick_fontsize", {}):
            for lbl in axd[mkey].get_xticklabels():
                lbl.set_fontsize(m["xtick_fontsize"][mkey])
        leg_loc = m.get("legend_loc", {}).get(mkey)
        if leg_loc and axd[mkey].get_legend() is not None:
            axd[mkey].get_legend().set_loc(leg_loc)
        hide_leg = m.get("hide_legend")
        if (hide_leg is True or mkey in (hide_leg or [])) \
                and axd[mkey].get_legend() is not None:
            axd[mkey].get_legend().remove()
        hide_tk = m.get("hide_ticks")
        if hide_tk is True or mkey in (hide_tk or []):
            axd[mkey].set_xticks([])
            axd[mkey].set_yticks([])
        if m.get("panel_labels", True):
            px = m.get("panel_label_x", {}).get(mkey)
            params = ({**PLOT_PARAMS, "panel_label_x": px} if px is not None
                      else PLOT_PARAMS)
            add_panel_label(axd[mkey], mkey, params=params)
    _share_axis_limits(axd, m)
    if m.get("suptitle"):
        fig.suptitle(
            m["suptitle"].format(exp_name=exp_name),
            fontsize=PLOT_PARAMS["suptitle_fontsize"], fontweight="bold",
        )
    # All mosaics are collected together in the shared <OUT_ROOT>/mosaics/ dir.
    save_fig(fig, fig_path("mosaics", name), dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return name
