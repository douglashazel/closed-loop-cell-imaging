"""Single source of truth for preprint figure style, sizing, and typography."""
import matplotlib as mpl


# =============================================================================
# Locked primitives
# =============================================================================
# Final on-page widths in inches for a single-column Word page
WIDTH_FULL = 6.5          # full text width (default for all figures)
WIDTH_TWO_THIRDS = 4.33
WIDTH_HALF = 3.25         # small side panels

DPI = 600                 # 600 for line/plot-heavy figures (everything here)

# Point sizes
FONT_BASE = 8             # default text / annotations
FONT_TITLE = 9            # axes titles
FONT_AXIS_LABEL = 8       # x / y axis labels
FONT_TICK = 7             # tick labels
FONT_LEGEND = 7           # legend entries
FONT_LEGEND_LARGE = 8     # legends that need to read slightly larger
FONT_SUPTITLE = 10        # figure suptitle


# =============================================================================
# PLOT_PARAMS
# =============================================================================
PLOT_PARAMS = {
    # --- dimensions (single-column Word page) ---
    "width_full": WIDTH_FULL,
    "width_two_thirds": WIDTH_TWO_THIRDS,
    "width_half": WIDTH_HALF,
    # Default figure sizes at final width. Single-panel plots use "figsize";
    # the wider/stacked multi-panel trace figures start from "figsize_wide".
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
    "scatter_size": 4,
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

# Default standalone figure size (single axis at full width).
FIGSIZE_DEFAULT = PLOT_PARAMS["figsize"]

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


# rcParams that must be identical for every figure
_RC = {
    "font.family": "sans-serif",
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

    "figure.constrained_layout.use": True,

    "savefig.dpi": DPI,

    "svg.fonttype": "none",     # keep SVG text as text
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """Apply the shared rcParams. Call once at the top of the plotting run."""
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


def add_mosaic_panel_labels(axd, letters=None, params=PLOT_PARAMS):
    """Letter every cell of a ``subplot_mosaic`` dict (a, b, c, ...).

    ``axd`` is the ``{label: ax}`` mapping returned by ``plt.subplot_mosaic``.
    By default each cell is labelled with its own mosaic key; pass ``letters``
    (a ``{mosaic_key: letter}`` map) to override (e.g. when mosaic keys are not
    already the panel letters you want printed).
    """
    for key, ax in axd.items():
        add_panel_label(ax, (letters or {}).get(key, key), params=params)
