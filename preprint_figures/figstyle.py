"""
Single source of truth for preprint figure style, sizing, and saving.

Import this in every figure script:

    from figstyle import PLOT_PARAMS, apply_style, add_panel_label
    apply_style()

WHY THIS EXISTS
---------------
Every figure is generated at the *exact* width it will occupy on the Word
page, and all font/line sizes are locked here in points. Matplotlib sizes are
in points, so a figure built at its final inch width shows its 8 pt labels as
8 pt on the page. Two figures at different widths still share the same point
sizes, so they match. The rule that follows is the important one:

    NEVER rescale a generated figure afterward (in Word or Inkscape).

Scaling shrinks the text with it, which is what breaks consistency. Insert the
PNG at 100%; if Word inserts at the wrong size, type the exact inches in the
Size dialog.

This module is the ONE place to change font, size, and line width. The legacy
``common/plot_params.py`` is now a thin re-export of the dicts defined here, so
the 8 analysis scripts keep importing ``from common.plot_params import
PLOT_PARAMS`` unchanged while their values are governed from here.

OPERATING RULES (encoded below, applied in every figure script)
---------------------------------------------------------------
* Generate at final width. Full Word text width (Letter, 1 in margins) ~6.5 in.
  Use ``PLOT_PARAMS['width_full']`` (6.5) by default; ``width_half`` (3.25) for
  small side panels.
* Do NOT use ``bbox_inches='tight'`` (it crops to content and changes the
  size). ``apply_style()`` turns on constrained layout globally so nothing
  clips inside the fixed size; do not also call ``plt.tight_layout()``.
* Deliver PNG at 600 dpi to Word (dpi is embedded so Word places it at the
  correct physical size) and keep the SVG as the editable Inkscape archive.
  ``common.io_paths.save_fig`` already writes both (PNG + sibling ``svg/``).
* Panel letters use Arial bold 11 pt; use the SAME spec in Inkscape so mixed
  figures match.
"""
import os
import matplotlib as mpl


# =============================================================================
# Locked primitives — change a number here, every figure follows.
# =============================================================================
# Final on-page widths in inches for a single-column Word page
# (Letter, 1 in margins -> ~6.5 in text width). Generate every figure at the
# width it will occupy, then insert at 100%.
WIDTH_FULL = 6.5          # full text width (default for all figures)
WIDTH_TWO_THIRDS = 4.33
WIDTH_HALF = 3.25         # small side panels

DPI = 600                 # 600 for line/plot-heavy figures (everything here)

# Point sizes — locked. These are wired into rcParams by apply_style() AND
# exposed under the legacy key names the scripts pass explicitly
# (fontsize=PLOT_PARAMS["title_fontsize"], ...), so there is one number each.
FONT_BASE = 8             # default text / annotations
FONT_TITLE = 9            # axes titles
FONT_AXIS_LABEL = 8       # x / y axis labels
FONT_TICK = 7             # tick labels
FONT_LEGEND = 7           # legend entries
FONT_LEGEND_LARGE = 8     # legends that need to read slightly larger
FONT_SUPTITLE = 10        # figure suptitle


# =============================================================================
# PLOT_PARAMS — merged dict. Dimensions + locked fonts + the full artist
# palette the analysis scripts reference by key. (Re-exported by
# common/plot_params.py for backward compatibility.)
# =============================================================================
PLOT_PARAMS = {
    # --- dimensions (single-column Word page) ---
    "width_full": WIDTH_FULL,
    "width_two_thirds": WIDTH_TWO_THIRDS,
    "width_half": WIDTH_HALF,
    # Default figure sizes at final width. Single-panel plots use "figsize";
    # the wider/stacked multi-panel trace figures start from "figsize_wide".
    # Per-figure scripts override the height where a layout needs more room.
    "figsize": (WIDTH_FULL, 3.9),
    "figsize_wide": (WIDTH_FULL, 4.2),
    "dpi": DPI,

    # --- locked fonts (legacy key names kept so per-artist calls still work) ---
    "base_fontsize": FONT_BASE,
    "title_fontsize": FONT_TITLE,
    "title_fontweight": "bold",
    "axis_label_fontsize": FONT_AXIS_LABEL,
    "tick_fontsize": FONT_TICK,
    "legend_fontsize": FONT_LEGEND,
    "legend_fontsize_large": FONT_LEGEND_LARGE,
    "suptitle_fontsize": FONT_SUPTITLE,

    # --- panel letters (a, b, c, d). Use the SAME spec in Inkscape. ---
    "panel_label_size": 11,
    "panel_label_weight": "bold",
    "panel_label_x": -0.12,     # axes-fraction offset; nudge per layout if clipped
    "panel_label_y": 1.12,

    # --- categorical palette ---
    "colors": ["#e74c3c", "#363fe9", "#e67e22", "#1a9d51"],
    # Muted, low-pop palette for the corr-vs-distance scatter clouds.
    "corr_scatter_colors": ["#e4776b", "#7fb0d1", "#f0984c", "#2b8a43"],
    "corr_fit_color": "#000000",   # black trend line
    "corr_band_color": "#9a9a9a",  # gray +/-3 SEM band

    # --- trace / image styling (line widths tuned for the 6.5 in final size) ---
    "cell_color": "#074f79cc",
    "cell_alpha": 0.3,
    "cell_lw": 0.4,
    "mean_color": "#1a1a1a",
    "mean_lw": 1.4,
    "stim_color": "#e74c3c",
    "stim_lw": 1.1,
    "f0_color": "#1a9d51",
    "f0_lw": 1.1,
    "trace_cmap": "twilight_shifted",
    "bg_cmap": "viridis",
    "img_cmap": "gray",
    "roi_color": "red",
    "roi_lw": 1.5,

    # --- violins + jittered points ---
    "violin_face": "#a0c8f0",
    "violin_edge": "#3782d3",
    "median_color": "#1aa821",
    "mean_marker_color": "#ed0d0d",
    "scatter_color": "#222222",
    "scatter_alpha": 0.5,
    "scatter_size": 12,
    "responder_color": "#8e44ad",     # responder scatter highlight
    "responder_edge": "#5a0000",      # dark red responder marker edge
    "jitter_strength": 0.08,
    "fit_color": "#363fe9",
    "pooled_mean_color": "#4a235a",   # dark purple pooled mean line
    "pooled_sem_color": "#8e44ad",    # purple +/-1 SEM band
    "pca_scatter_color": "#1a5e1a",   # dark green PCA/UMAP scatter
    "rr_color": "#363fe9",            # blue responder x responder pairs
    # Per-replicate (per-channel) train-mean inset — green ramp dark->light.
    "replicate_greens": ["#1b5e20", "#43a047", "#a5d6a7"],
}

# Used by: sliding-window correlation
PLOT_PARAMS_SLIDING = {
    "figsize": (WIDTH_FULL, 3.0),
    "dpi": DPI,
    "title_fontsize": FONT_TITLE,
    "title_fontweight": "bold",
    "suptitle_fontsize": FONT_SUPTITLE,
    "axis_label_fontsize": FONT_AXIS_LABEL,
    "legend_fontsize": FONT_LEGEND,
    "window_size": 30,           # frames per sliding window
    "step": 15,                  # frames between window centers
    "global_corr_cutoff": 0.6,   # exclude pairs with full-series Pearson >= this
    "line_alpha": 0.04,          # individual pair lines
    "line_lw": 0.4,
    "mean_lw": 2.0,
    "sem_alpha": 0.18,
    "sem_n": 6,                  # number of SEMs to shade
    "pearson_color": "#0b95e5",
    "spearman_color": "#dc2846",
    "mean_color_pearson": "#003d6b",
    "mean_color_spearman": "#7a0020",
    "stim_color": "#2a8618",
    "stim_lw": 1.4,
}

# Used by: NRK hardware feedback luminosity log
PLOT_PARAMS_HW_LOG = {
    "figsize": (WIDTH_FULL, 2.8),
    "dpi": DPI,
    "title_fontsize": FONT_TITLE,
    "title_fontweight": "bold",
    "axis_label_fontsize": FONT_AXIS_LABEL,
    "legend_fontsize": FONT_LEGEND,
    "line_color": "steelblue",
    "line_lw": 1.2,
    "acid_color": "#c0392b",
    "acid_lw": 0.8,
    "setpoint_colors": ["#e67e22", "#1a9d51", "#9b59b6", "#3498db", "#f1c40f"],
    "setpoint_alpha": 0.22,
    "setpoint_lw": 1.1,
}


# rcParams that must be identical for every figure. Sizes are in points, so a
# figure generated at its final inch width shows these exact point sizes on the
# page; two figures at different widths still share the same text size.
_RC = {
    "font.family": "sans-serif",
    # Arial first so figures auto-upgrade to true Arial if it is ever installed.
    # Arial is NOT installed in this environment, so the effective font is
    # Liberation Sans -- the metric-compatible Arial substitute (identical glyph
    # widths, near-identical appearance). Nimbus Sans is a second Helvetica/Arial
    # -alike fallback; DejaVu Sans is the last resort. To use real Arial, drop
    # Arial.ttf into ~/.local/share/fonts/ (or apt install ttf-mscorefonts-
    # installer) and clear the cache: rm -rf ~/.cache/matplotlib, then rerun.
    "font.sans-serif": ["Arial", "Liberation Sans", "Nimbus Sans",
                        "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",

    "font.size": FONT_BASE,
    "axes.titlesize": FONT_TITLE,
    "axes.titleweight": "bold",
    "axes.labelsize": FONT_AXIS_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "figure.titlesize": FONT_SUPTITLE,
    "figure.titleweight": "bold",

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,

    "axes.spines.top": False,
    "axes.spines.right": False,

    # Generate at a fixed size; let constrained layout pack content inside it
    # (NOT tight_layout / bbox_inches='tight', which change the size).
    "figure.constrained_layout.use": True,

    "savefig.dpi": DPI,

    # Vector output that stays portable and editable in Inkscape.
    "svg.fonttype": "none",     # keep SVG text as text
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """Apply the shared rcParams. Call once at the top of each figure script."""
    mpl.rcParams.update(_RC)


def add_panel_label(ax, label, params=PLOT_PARAMS):
    """Add a bold panel letter (a, b, ...) at the top-left of an axes."""
    ax.text(
        params["panel_label_x"], params["panel_label_y"], label,
        transform=ax.transAxes,
        fontsize=params["panel_label_size"],
        fontweight=params["panel_label_weight"],
        va="top", ha="right",
    )


def save_figure(fig, stem, outdir="figures", dpi=None):
    """
    Save a figure at its EXACT figsize (no tight bbox) as PNG + SVG.

    Convenience saver for standalone / template figures. The production
    analysis scripts save into ``April28_preprint_results/<experiment>/`` via
    ``common.io_paths.save_fig`` (which also writes a sibling ``svg/``); both
    paths obey the same rule — no ``bbox_inches='tight'``.

    PNG goes into Word; the embedded dpi makes Word place it at the correct
    physical size, so insert at 100% and do not drag-resize. SVG is the
    editable archive and the source for Inkscape assembly.
    """
    if dpi is None:
        dpi = PLOT_PARAMS["dpi"]
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{stem}.png"), dpi=dpi)  # NOT bbox_inches='tight'
    fig.savefig(os.path.join(outdir, f"{stem}.svg"))
