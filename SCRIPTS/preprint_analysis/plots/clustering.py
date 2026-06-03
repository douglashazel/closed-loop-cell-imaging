"""Render functions for the clustering figures (reads analysis_cache/<exp>/clustering.pkl).

The old 1×2 ``pooled_pca_umap_uncolored`` figure (PCA | UMAP) is decomposed into
two standalone single-axis figures — ``pooled_pca_only`` (PC1 vs PC2) and
``pooled_umap_only`` (UMAP1 vs UMAP2). Both draw the pooled-across-channels
embedding with every point one color (no clustering). Label/title/suptitle text
lives centrally in ``figures_spec.py``; these functions pull it from the passed
``spec``. No recompute happens here — the cached PCA scores and frozen UMAP
embedding are plotted directly.
"""
from plots._base import (
    PLOT_PARAMS,
    clean_axes,
    title_of,
    xlabel_of,
    ylabel_of,
)

NAME = "clustering"


def _scatter(ax, x, y, spec, fill):
    """Shared one-color scatter (PCA or UMAP), styled exactly as the source."""
    P = PLOT_PARAMS
    clean_axes(ax)
    ax.scatter(
        x, y,
        s=P["scatter_size"] * 1.4,
        color=P["pca_scatter_color"],
        alpha=P["scatter_alpha"],
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel(xlabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_ylabel(ylabel_of(spec, fill), fontsize=P["axis_label_fontsize"])
    ax.set_title(title_of(spec, fill), fontsize=P["title_fontsize"],
                 fontweight=P["title_fontweight"])


def render_pca_only(ax, payload, spec, *, fill):
    """PCA scatter: PC1 vs PC2 (pooled across channels, no clustering)."""
    pcs = payload["pcs"]
    _scatter(ax, pcs[:, 0], pcs[:, 1], spec, fill)


def render_umap_only(ax, payload, spec, *, fill):
    """UMAP scatter: UMAP1 vs UMAP2 (the frozen, cached embedding)."""
    embedding = payload["embedding"]
    _scatter(ax, embedding[:, 0], embedding[:, 1], spec, fill)


def iter_figures(blob, exp_name):
    """Yield ``(spec_key, payload, fill)`` for the two clustering figures."""
    data = blob["data"]
    meta = blob["meta"]

    fill = {
        "exp_name": exp_name,
        "ch_label": meta["ch_label"],
        "n_cells": meta["n_cells"],
        "evr0_pct": meta["evr0_pct"],
        # evr1_pct is None when only one PC; the PCA ylabel falls back to "PC 2"
        # (selected via the n_pca>1 branch in figures_spec), so the template is
        # only formatted with evr1_pct when it is present.
        "evr1_pct": (0.0 if meta.get("evr1_pct") is None else meta["evr1_pct"]),
    }

    yield ("pooled_pca_only", {"pcs": data["pcs"]}, dict(fill))
    yield ("pooled_umap_only", {"embedding": data["embedding"]}, dict(fill))
