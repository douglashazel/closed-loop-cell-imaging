#!/usr/bin/env python3
"""Trace embedding: PCA + UMAP scatter (no clustering).

Per experiment:
    * pooled_pca_umap_uncolored.png   — PCA + UMAP scatter pooled across channels
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import fig_path, save_fig
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.time_axis import frames_to_min
from figstyle import apply_style

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402

apply_style()


def _compute_clustering_embeddings(state, exp_name, ch, random_state=0,
                                   n_pca_max=20, n_neighbors=15):
    """Z-score per cell, run PCA + UMAP. Returns None if too few cells."""
    pooled = isinstance(ch, (list, tuple))
    channels = list(ch) if pooled else [ch]
    ref_ch = channels[0]

    df_ref = lum_dict_to_df(
        state["corrected_lum"][exp_name][ref_ch]
    ).set_index("CellID")
    ref_frame_cols = sorted(
        [c for c in df_ref.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    frame_nums = np.array([int(str(c).lstrip("f")) for c in ref_frame_cols])
    frame_min = frames_to_min(state, exp_name, ref_ch, frame_nums)

    per_channel_cols = []
    for c in channels:
        df_c = lum_dict_to_df(
            state["corrected_lum"][exp_name][c]
        ).set_index("CellID")
        cols_c = sorted(
            [k for k in df_c.columns if str(k).startswith("f")],
            key=lambda k: int(str(k).lstrip("f")),
        )
        per_channel_cols.append((df_c, cols_c, c))
    n_common = min(len(cols_c) for _, cols_c, _ in per_channel_cols)
    if pooled and any(len(cols_c) != n_common for _, cols_c, _ in per_channel_cols):
        print(
            f"  {exp_name}: pooled clustering — channel frame counts "
            f"{[len(cols_c) for _, cols_c, _ in per_channel_cols]} differ; "
            f"truncating to common length {n_common}."
        )
        frame_nums = frame_nums[:n_common]
        frame_min = frame_min[:n_common]
    matrices = []
    cell_ids_initial = []
    for df_c, cols_c, c in per_channel_cols:
        matrices.append(df_c[cols_c[:n_common]].values.astype(float))
        cell_ids_initial.extend((c, cid) for cid in df_c.index.tolist())
    X_raw = np.vstack(matrices)
    row_all_nan = np.isnan(X_raw).all(axis=1)
    X_raw = X_raw[~row_all_nan]
    cell_ids_after_nan = [
        cid for cid, drop in zip(cell_ids_initial, row_all_nan) if not drop
    ]
    if X_raw.shape[0] < 5:
        return None
    row_means = np.nanmean(X_raw, axis=1, keepdims=True)
    X_raw = np.where(np.isnan(X_raw), row_means, X_raw)

    mu = X_raw.mean(axis=1, keepdims=True)
    sd = X_raw.std(axis=1, keepdims=True)
    keep = sd[:, 0] > 0
    X_raw = X_raw[keep]
    mu = mu[keep]
    sd = sd[keep]
    cell_ids_aligned = [
        cid for cid, kept in zip(cell_ids_after_nan, keep.tolist()) if kept
    ]
    X_z = (X_raw - mu) / sd
    X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

    n_cells, n_frames = X_z.shape
    if n_cells < 5:
        return None

    n_pca = int(min(n_pca_max, n_cells, n_frames))
    pca = PCA(n_components=n_pca, random_state=random_state)
    pcs = pca.fit_transform(X_z)
    evr = pca.explained_variance_ratio_

    nn = int(min(n_neighbors, max(2, n_cells - 1)))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=0.1,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(pcs)

    return {
        "X_raw": X_raw,
        "frame_nums": frame_nums,
        "frame_min": frame_min,
        "pcs": pcs,
        "evr": evr,
        "embedding": embedding,
        "n_pca": n_pca,
        "n_cells": n_cells,
        "ref_ch": ref_ch,
        "cell_ids": cell_ids_aligned,
    }


def _render_pca_umap_uncolored(
    state, exp_name, ch_arg, ch_label, save_name, *, random_state,
):
    """1×2 figure: PC1 vs PC2 + UMAP, all points one color (no clustering)."""
    emb = _compute_clustering_embeddings(
        state, exp_name, ch_arg, random_state=random_state,
    )
    if emb is None:
        print(f"{exp_name} / {ch_label}: too few cells — skipping uncolored PCA/UMAP")
        return
    pcs = emb["pcs"]
    evr = emb["evr"]
    embedding = emb["embedding"]
    n_cells = emb["n_cells"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(PLOT_PARAMS["width_full"], 3.0),
        dpi=PLOT_PARAMS["dpi"],
    )
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)

    ax = axes[0]
    ax.scatter(
        pcs[:, 0], pcs[:, 1],
        s=PLOT_PARAMS["scatter_size"] * 1.4,
        color=PLOT_PARAMS["pca_scatter_color"],
        alpha=PLOT_PARAMS["scatter_alpha"],
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel(
        f"PC 1 ({evr[0] * 100:.1f}% var)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_ylabel(
        f"PC 2 ({evr[1] * 100:.1f}% var)" if len(evr) > 1 else "PC 2",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_title(
        "PCA: PC1 vs PC2",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )

    ax = axes[1]
    ax.scatter(
        embedding[:, 0], embedding[:, 1],
        s=PLOT_PARAMS["scatter_size"] * 1.4,
        color=PLOT_PARAMS["pca_scatter_color"],
        alpha=PLOT_PARAMS["scatter_alpha"],
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel("UMAP 1", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("UMAP 2", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        "UMAP embedding",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )

    fig.suptitle(
        f"{exp_name} / {ch_label} — PCA + UMAP (no clustering, n={n_cells})",
        fontsize=PLOT_PARAMS["title_fontsize"] + 1,
        fontweight="bold",
    )
    save_fig(
        fig, fig_path(exp_name, save_name),
        dpi=PLOT_PARAMS["dpi"],
    )
    plt.close(fig)


def plot_pca_umap_uncolored(experiments, state, *, random_state=0):
    """PCA + UMAP scatters pooled across channels (all points one color)."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        if not channels:
            continue
        _render_pca_umap_uncolored(
            state, exp_name, channels,
            f"pooled ({', '.join(channels)})",
            "pooled_pca_umap_uncolored",
            random_state=random_state,
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_pca_umap_uncolored(experiments, state)


if __name__ == "__main__":
    main()
