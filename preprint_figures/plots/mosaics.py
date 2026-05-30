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


def build_mosaic(name, exp_name):
    """Assemble mosaic ``name`` for one experiment and save the PNG."""
    if name not in MOSAICS:
        raise KeyError(f"unknown mosaic '{name}'. Known: {list(MOSAICS)}")
    m = MOSAICS[name]
    apply_style()
    fig, axd = plt.subplot_mosaic(
        m["layout"], figsize=m["figsize"], layout="constrained",
    )
    for mkey, (spec_key, match) in m["cells"].items():
        spec, payload, fill = _find_instance(exp_name, spec_key, match)
        if spec.multi_panel:
            raise ValueError(
                f"mosaic cell '{mkey}' references multi-panel figure "
                f"'{spec_key}'; compose those via fig.subfigures instead."
            )
        spec.render(axd[mkey], payload, spec, fill=fill)
        add_panel_label(axd[mkey], mkey)
    if m.get("suptitle"):
        fig.suptitle(
            m["suptitle"].format(exp_name=exp_name),
            fontsize=PLOT_PARAMS["suptitle_fontsize"], fontweight="bold",
        )
    save_fig(fig, fig_path(exp_name, name), dpi=PLOT_PARAMS["dpi"])
    plt.close(fig)
    return name
