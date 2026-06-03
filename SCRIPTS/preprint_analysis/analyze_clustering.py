#!/usr/bin/env python3
"""Clustering analysis (no plotting) → analysis_cache/<exp>/clustering.pkl.

Computes the figure-ready intermediates the clustering figures display: a
pooled-across-channels PCA + UMAP embedding (no clustering / no coloring).

Per experiment, traces are z-scored per cell, then PCA (same ``n_components``
logic as the source script) and UMAP (``random_state=0``, same
``n_neighbors``/``min_dist``) are fit. The 2-D UMAP embedding is cached so the
figures are frozen against the installed umap version.

The matplotlib rendering lives in ``plots/clustering.py``; this script writes
only numbers. The PCA/UMAP math is verbatim from the original ``clustering.py``.
"""

import os
import sys

import numpy as np
import umap
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import save_analysis_cache
from common.pipeline import prepare_state
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS/core_pipeline")
from io_utils import lum_dict_to_df  # noqa: E402


def _compute_clustering_embeddings(state, exp_name, ch, random_state=0,
                                   n_pca_max=20, n_neighbors=15):
    """Z-score per cell, run PCA + UMAP. Returns None if too few cells.

    Verbatim port of the source ``clustering.py`` math (minus the matplotlib).
    """
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
        "pcs": np.asarray(pcs, dtype=float),
        "evr": np.asarray(evr, dtype=float),
        "embedding": np.asarray(embedding, dtype=float),
        "n_pca": int(n_pca),
        "n_cells": int(n_cells),
        "ref_ch": ref_ch,
        "cell_ids": cell_ids_aligned,
        "channels": channels,
    }


def analyze(experiments, state, *, random_state=0):
    """Compute + cache one pooled PCA/UMAP bundle per experiment."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        if not channels:
            print(f"  {exp_name}: no channels — skipping clustering")
            continue
        ch_label = f"pooled ({', '.join(channels)})"
        emb = _compute_clustering_embeddings(
            state, exp_name, channels, random_state=random_state,
        )
        if emb is None:
            print(
                f"  {exp_name} / {ch_label}: too few cells — "
                f"skipping clustering cache"
            )
            continue

        evr = emb["evr"]
        meta = {
            "exp_name": exp_name,
            "ch_label": ch_label,
            "n_cells": int(emb["n_cells"]),
            "n_pca": int(emb["n_pca"]),
            "evr0_pct": float(evr[0] * 100) if len(evr) > 0 else 0.0,
            "evr1_pct": float(evr[1] * 100) if len(evr) > 1 else None,
        }
        save_analysis_cache(emb, exp_name, "clustering", meta=meta)
        print(
            f"  cached clustering.pkl for {exp_name} "
            f"(n={emb['n_cells']}, n_pca={emb['n_pca']})"
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
