"""Plot styling dicts copied verbatim from april28_final_figures.py."""

# Used by: bg diagnostic, time traces, dF/F0, corr-vs-dist, violin
PLOT_PARAMS = {
    "figsize": (10, 6),
    "figsize_wide": (18, 10),
    "dpi": 300,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "axis_label_fontsize": 13,
    "legend_fontsize": 10,
    "legend_fontsize_large": 14,
    "colors": ["#e74c3c", "#363fe9", "#e67e22", "#1a9d51"],
    # Muted, low-pop palette for the corr-vs-distance scatter clouds.
    # Earth-tone desaturations: slate, taupe, sage, mauve.
    "corr_scatter_colors": ["#e4776b", "#7fb0d1", "#f0984c", "#2b8a43"],
    "corr_fit_color": "#000000",   # black trend line
    "corr_band_color": "#9a9a9a",  # gray ±3 SEM band
    "cell_color": "#074f79cc",
    "cell_alpha": 0.3,
    "cell_lw": 0.5,
    "mean_color": "#1a1a1a",
    "mean_lw": 1.8,
    "stim_color": "#e74c3c",
    "stim_lw": 1.5,
    "f0_color": "#1a9d51",
    "f0_lw": 1.5,
    "trace_cmap": "twilight_shifted",
    "bg_cmap": "viridis",
    "img_cmap": "gray",
    "roi_color": "red",
    "roi_lw": 2.0,
    "violin_face": "#a0c8f0",
    "violin_edge": "#3782d3",
    "median_color": "#1aa821",
    "mean_marker_color": "#ed0d0d",
    "scatter_color": "#222222",
    "scatter_alpha": 0.5,
    "scatter_size": 12,
    "jitter_strength": 0.08,
    "fit_color": "#363fe9",
    "pooled_mean_color": "#4a235a",   # dark purple — pooled mean line
    "pooled_sem_color": "#8e44ad",    # purple — ±1 SEM band
    "pca_scatter_color": "#1a5e1a",   # dark green — PCA/UMAP scatter
    "rr_color": "#363fe9",            # blue — responder × responder pairs
    # Per-replicate (per-channel) train-mean inset — green ramp dark→light.
    "replicate_greens": ["#1b5e20", "#43a047", "#a5d6a7"],
}

# Used by: sliding-window correlation
PLOT_PARAMS_SLIDING = {
    "figsize": (18, 7),
    "dpi": 300,
    "title_fontsize": 13,
    "title_fontweight": "bold",
    "suptitle_fontsize": 15,
    "axis_label_fontsize": 13,
    "legend_fontsize": 10,
    "window_size": 30,           # frames per sliding window
    "step": 15,                  # frames between window centers
    "global_corr_cutoff": 0.6,   # exclude pairs with full-series Pearson >= this
    "line_alpha": 0.04,          # individual pair lines
    "line_lw": 0.4,
    "mean_lw": 2.5,
    "sem_alpha": 0.18,
    "sem_n": 6,                  # number of SEMs to shade
    "pearson_color": "#0b95e5",
    "spearman_color": "#dc2846",
    "mean_color_pearson": "#003d6b",
    "mean_color_spearman": "#7a0020",
    "stim_color": "#2a8618",
    "stim_lw": 1.8,
}

# Used by: NRK hardware feedback luminosity log
PLOT_PARAMS_HW_LOG = {
    "figsize": (12, 5),
    "dpi": 300,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "axis_label_fontsize": 12,
    "legend_fontsize": 9,
    "line_color": "steelblue",
    "line_lw": 1.4,
    "acid_color": "#c0392b",
    "acid_lw": 0.9,
    "setpoint_colors": ["#e67e22", "#1a9d51", "#9b59b6", "#3498db", "#f1c40f"],
    "setpoint_alpha": 0.22,
    "setpoint_lw": 1.2,
}
