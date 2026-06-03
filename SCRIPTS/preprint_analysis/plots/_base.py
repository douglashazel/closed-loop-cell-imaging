"""Render-function contract + drivers shared by every ``plots/*`` module.

A render function draws ONE figure's content. Single-axis figures draw onto a
passed-in ``Axes`` so the SAME function serves both a standalone figure and a
``subplot_mosaic`` cell (identical fontsizes). The four ``responder_diagnostic``
figures are the documented exception: they draw onto a passed-in
``Figure``/``SubFigure`` and build their own gridspec.

    def render_xxx(ax, payload, spec, *, fill):    # single-axis
        ...                                         # ax.plot / set_title / ...

    def render_yyy(fig, payload, spec, *, fill):    # multi-panel exception
        ...                                         # fig.subplots / gridspec

``payload`` is the cache slice the figure needs; ``spec`` is the ``FigureSpec``
(label/title/legend text templates, defined centrally in ``figures_spec.py``);
``fill`` is the dict of runtime values used to fill those templates (always
includes ``"exp_name"``, which also selects the output directory).

Render functions NEVER call ``plt.subplots``/``savefig``/``suptitle`` (the
driver owns the figure). They set their own axes ``title``/``xlabel``/``ylabel``
/``legend`` from ``spec`` via the ``fmt`` helper.
"""
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve sibling top-level modules (style, common) regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.io_paths import fig_path, save_fig  # noqa: E402
from style import PLOT_PARAMS, apply_style, FIGSIZE_DEFAULT  # noqa: E402


# =============================================================================
# FigureSpec — the central label/title/legend record (instances live in
# figures_spec.py). Templates use str.format placeholders filled from `fill`.
# =============================================================================
@dataclass(frozen=True)
class FigureSpec:
    id: str                       # filename-stem template, e.g. "{ch}_dff_raw{subset_suffix}"
    analysis: str                 # which analysis cache it reads ("dff", ...)
    render: Callable              # render_*(ax_or_fig, payload, spec, *, fill)
    scope: str = "experiment"     # experiment | channel | channel_subset | cross_experiment
    multi_panel: bool = False     # True only for the responder_diagnostic figures
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    suptitle: str = ""            # standalone-only figure suptitle (skipped in mosaics)
    legend: Optional[dict] = None  # legend label templates, e.g. {"mean": "Mean"}
    caveat: str = ""              # standalone-only footnote (e.g. inferential caveat)
    figsize: Optional[Tuple[float, float]] = None


# =============================================================================
# Small shared drawing helpers (kept here so the plot layer needs nothing from
# the analysis-only common.stim_helpers module).
# =============================================================================
def clean_axes(ax):
    """Hide the top/right spines and their ticks (the locked house style)."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)


def draw_stim_spans(ax, spans, label, color, alpha=0.18):
    """Shade each cached ``(start_min, end_min)`` span; label only the first."""
    for idx, (start_m, end_m) in enumerate(spans):
        ax.axvspan(
            start_m, end_m,
            color=color, alpha=alpha, linewidth=0, zorder=0,
            label=label if idx == 0 else None,
        )


def _reserve_bottom(fig, frac):
    """Shrink the constrained-layout area to leave a bottom band (figure
    fraction ``frac``) for a footnote. No-op if the engine lacks rect support."""
    eng = fig.get_layout_engine()
    if eng is None:
        return
    try:
        eng.set(rect=(0.0, frac, 1.0, 1.0 - frac))
    except (TypeError, AttributeError, ValueError):
        pass


def fmt(template, fill):
    """Format a label template with ``fill``; an empty template returns ''."""
    return template.format(**fill) if template else ""


def title_of(spec, fill):
    return fmt(spec.title, fill)


def xlabel_of(spec, fill):
    return fmt(spec.xlabel, fill)


def ylabel_of(spec, fill):
    return fmt(spec.ylabel, fill)


def legend_text(spec, key, fill):
    """Formatted legend label for ``spec.legend[key]`` (empty if missing)."""
    if not spec.legend or key not in spec.legend:
        return ""
    return fmt(spec.legend[key], fill)


# =============================================================================
# Drivers — build a standalone figure around a render function.
# =============================================================================
def make_standalone(spec, payload, fill):
    """Render a single-axis figure type and save it as PNG.

    Output goes to ``fig_path(fill["exp_name"], spec.id.format(**fill))``; for
    cross-experiment figures pass ``fill["exp_name"]`` = the output-bucket dir
    (e.g. ``"dmso_stim8_comparison"``).
    """
    apply_style()
    fig, ax = plt.subplots(
        figsize=spec.figsize or FIGSIZE_DEFAULT, dpi=PLOT_PARAMS["dpi"],
    )
    spec.render(ax, payload, spec, fill=fill)
    # Long titles wrap to the figure width instead of clipping at the fixed
    # size (we never use bbox_inches='tight', which would change the size).
    if ax.get_title():
        ax.title.set_wrap(True)
    if spec.suptitle:
        st = fig.suptitle(
            fmt(spec.suptitle, fill),
            fontsize=PLOT_PARAMS["suptitle_fontsize"], fontweight="bold",
        )
        st.set_wrap(True)
    if spec.caveat:
        # Reserve a bottom band so the footnote clears the x-axis label.
        _reserve_bottom(fig, 0.08)
        cv = fig.text(
            0.5, 0.012, fmt(spec.caveat, fill),
            ha="center", va="bottom",
            fontsize=PLOT_PARAMS["tick_fontsize"], color="#555555",
        )
        cv.set_wrap(True)
    stem = spec.id.format(**fill)
    save_fig(fig, fig_path(fill["exp_name"], stem), dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return stem


def make_standalone_fig(spec, payload, fill):
    """Render a multi-panel figure type (the responder_diagnostic exception).

    The render function receives the whole ``Figure`` and owns its gridspec,
    suptitle, and any ``set_layout_engine('none')`` it needs.
    """
    apply_style()
    fig = plt.figure(
        figsize=spec.figsize or PLOT_PARAMS["figsize_wide"],
        dpi=PLOT_PARAMS["dpi"],
    )
    spec.render(fig, payload, spec, fill=fill)
    stem = spec.id.format(**fill)
    save_fig(fig, fig_path(fill["exp_name"], stem), dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return stem


def render_one(spec, payload, fill):
    """Dispatch to the right driver based on ``spec.multi_panel``."""
    if spec.multi_panel:
        return make_standalone_fig(spec, payload, fill)
    return make_standalone(spec, payload, fill)
