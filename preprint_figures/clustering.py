#!/usr/bin/env python3
"""Trace clustering: PCA + UMAP + KMeans.

Per (experiment, channel):
    * <ch>_trace_clustering.png           — 3x2 panel (auto-k by silhouette)
    * <ch>_trace_clustering_kselect.png   — UMAP at k ∈ {2,3,4,5}
Per experiment:
    * pooled_trace_clustering.png         — pooled across channels
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.cluster_labels import save_cluster_labels
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stim_helpers import (
    compute_f0_baseline,
    draw_stim_spans,
    stim_spans_min,
    stim_timing_aligned_across_channels,
)
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


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


def _silhouette_sweep(pcs, k_min=2, k_max=8, random_state=0):
    """Fit KMeans for each k and return (ks, scores, best_k, best_score)."""
    n_cells = pcs.shape[0]
    k_upper = min(k_max, n_cells - 1)
    ks = []
    scores = []
    best_k = k_min
    best_score = -np.inf
    for k in range(k_min, k_upper + 1):
        km_try = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labs_try = km_try.fit_predict(pcs)
        if len(np.unique(labs_try)) < 2:
            continue
        try:
            s = silhouette_score(pcs, labs_try)
        except ValueError:
            continue
        ks.append(k)
        scores.append(s)
        if s > best_score:
            best_score = s
            best_k = k
    return np.array(ks), np.array(scores), best_k, best_score


def _render_trace_clustering(
    state, exp_name, cfg, ch_arg, ch_label, save_name,
    *, k_min, k_max, random_state,
):
    """Internal renderer used by both per-channel and pooled clustering."""
    emb = _compute_clustering_embeddings(
        state, exp_name, ch_arg, random_state=random_state,
    )
    if emb is None:
        print(
            f"{exp_name} / {ch_label}: too few cells — skipping clustering"
        )
        return
    pcs = emb["pcs"]
    evr = emb["evr"]
    embedding = emb["embedding"]
    X_raw = emb["X_raw"]
    frame_min = emb["frame_min"]
    n_cells = emb["n_cells"]
    n_pca = emb["n_pca"]
    ref_ch = emb["ref_ch"]
    cell_ids = emb["cell_ids"]

    ks, scores, best_k, best_score = _silhouette_sweep(
        pcs, k_min=k_min, k_max=k_max, random_state=random_state,
    )
    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels = km.fit_predict(pcs)

    cluster_cache_key = "__pooled__" if isinstance(ch_arg, (list, tuple)) else ch_arg
    save_cluster_labels(
        exp_name, cluster_cache_key,
        cell_ids=cell_ids,
        labels=labels,
        best_k=best_k,
        silhouette=float(best_score),
        k_method="silhouette_2_to_8",
    )

    cmap = plt.get_cmap("tab10")
    cluster_colors = [cmap(i % 10) for i in range(best_k)]

    fig, axes = plt.subplots(
        3, 2,
        figsize=(16, 14),
        dpi=PLOT_PARAMS["dpi"],
    )
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)

    ax = axes[0, 0]
    n_show = int(min(10, len(evr)))
    ax.bar(
        np.arange(1, n_show + 1), evr[:n_show],
        color=PLOT_PARAMS["fit_color"], alpha=0.85,
    )
    ax.set_xlabel("Principal component", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Explained variance ratio", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_xticks(np.arange(1, n_show + 1))
    ax.set_title(
        f"PCA scree (top {n_show} of {n_pca}) — cum. {evr[:n_show].sum() * 100:.1f}%",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )

    ax = axes[0, 1]
    if len(ks):
        ax.plot(
            ks, scores,
            "-o",
            color=PLOT_PARAMS["fit_color"],
            linewidth=1.8,
            markersize=7,
        )
        ax.plot(
            [best_k], [best_score],
            marker="*",
            color=PLOT_PARAMS["mean_marker_color"],
            markersize=20,
            markeredgecolor="#222222",
            markeredgewidth=0.8,
            linestyle="None",
            label=f"chosen k={best_k} (s={best_score:.3f})",
            zorder=5,
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    ax.set_xlabel("k (number of clusters)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Silhouette score", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_xticks(ks if len(ks) else [k_min])
    ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.set_title(
        f"Silhouette across k=[{k_min}, {min(k_max, n_cells - 1)}]",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )

    ax = axes[1, 0]
    for cid in range(best_k):
        mask = labels == cid
        ax.scatter(
            pcs[mask, 0], pcs[mask, 1],
            s=PLOT_PARAMS["scatter_size"] * 1.4,
            color=cluster_colors[cid],
            alpha=0.75,
            edgecolors="none",
            label=f"Cluster {cid} (n={int(mask.sum())})",
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
        "PCA: PC1 vs PC2 (interpretable axes)",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

    ax = axes[1, 1]
    for cid in range(best_k):
        mask = labels == cid
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=PLOT_PARAMS["scatter_size"] * 1.4,
            color=cluster_colors[cid],
            alpha=0.75,
            edgecolors="none",
            label=f"Cluster {cid} (n={int(mask.sum())})",
        )
    ax.set_xlabel("UMAP 1", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("UMAP 2", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        f"UMAP embedding — k={best_k} (silhouette={best_score:.3f})",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

    ax = axes[2, 0]
    pooled_misaligned = (
        isinstance(ch_arg, list)
        and not stim_timing_aligned_across_channels(cfg, state, exp_name)
    )
    if not pooled_misaligned:
        spans, stim_label = stim_spans_min(state, exp_name, ref_ch, cfg)
    else:
        spans, stim_label = [], None
    _, _, first_stim = compute_f0_baseline(state, exp_name, ref_ch, cfg)
    baseline_lo_min = frames_to_min(state, exp_name, ref_ch, [0])[0]
    baseline_hi_min = frames_to_min(
        state, exp_name, ref_ch, [max(first_stim - 1, 0)]
    )[0]
    baseline_label = (
        f"F₀ baseline (frames 0–{first_stim - 1})"
        if first_stim > 1
        else "F₀ baseline (frame 0)"
    )
    for cid in range(best_k):
        mask = labels == cid
        cluster_traces = X_raw[mask]
        mean_trace = np.nanmean(cluster_traces, axis=0)
        n_in = max(int(mask.sum()), 1)
        sem_trace = np.nanstd(cluster_traces, axis=0) / np.sqrt(n_in)
        ax.fill_between(
            frame_min, mean_trace - sem_trace, mean_trace + sem_trace,
            color=cluster_colors[cid], alpha=0.15, linewidth=0,
        )
        ax.plot(
            frame_min, mean_trace,
            color=cluster_colors[cid],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=f"Cluster {cid} (n={n_in})",
        )
    draw_stim_spans(
        ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18,
    )
    ax.axvspan(
        baseline_lo_min, baseline_hi_min,
        color=PLOT_PARAMS["f0_color"], alpha=0.10, zorder=0,
        label=baseline_label,
    )
    rsp = state["real_setpoint_min"][exp_name].get(ref_ch)
    if rsp is not None:
        ax.axvline(
            rsp,
            color="#000000",
            linewidth=2.0, linestyle=":", alpha=0.9,
            label=f"Real setpoint ({rsp:.1f} min)",
        )
    ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel(
        "Corrected luminosity (cluster mean ±1 SEM)",
        fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_title(
        "Cluster mean traces",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")

    ax = axes[2, 1]
    sizes = np.array([int((labels == cid).sum()) for cid in range(best_k)])
    ax.bar(
        np.arange(best_k), sizes,
        color=cluster_colors, edgecolor="#222222", linewidth=0.6,
    )
    for cid, n in enumerate(sizes):
        ax.text(
            cid, n, str(int(n)),
            ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"],
        )
    ax.set_xticks(np.arange(best_k))
    ax.set_xticklabels([f"C{cid}" for cid in range(best_k)])
    ax.set_xlabel("Cluster", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        f"Cluster sizes (total {n_cells})",
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )

    fig.suptitle(
        f"{exp_name} / {ch_label} — trace clustering (k={best_k}, n={n_cells})",
        fontsize=PLOT_PARAMS["title_fontsize"] + 1,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    fig.savefig(
        fig_path(exp_name, save_name),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)


def plot_trace_clustering(
    experiments, state, k_min=2, k_max=8, random_state=0,
    *, pool_channels=False,
):
    """Cluster cells by trace shape; visualize as a 3x2 figure per channel."""
    for exp_name, cfg in experiments.items():
        if pool_channels:
            channels = cfg["channels"]
            if len(channels) < 1:
                continue
            ch_arg = channels
            ch_label = f"pooled ({', '.join(channels)})"
            save_name = "pooled_trace_clustering"
        else:
            ch_arg_list = [(ch, ch, f"{ch}_trace_clustering") for ch in cfg["channels"]]
            for ch, ch_label, save_name in ch_arg_list:
                _render_trace_clustering(
                    state, exp_name, cfg, ch, ch, save_name,
                    k_min=k_min, k_max=k_max, random_state=random_state,
                )
            continue
        _render_trace_clustering(
            state, exp_name, cfg, ch_arg, ch_label, save_name,
            k_min=k_min, k_max=k_max, random_state=random_state,
        )


def plot_trace_clustering_kselect(experiments, state, ks=(2, 3, 4, 5),
                                   random_state=0):
    """Per (exp, ch): a 1×len(ks) row of UMAP scatters at fixed k values."""
    cmap = plt.get_cmap("tab10")
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            emb = _compute_clustering_embeddings(
                state, exp_name, ch, random_state=random_state,
            )
            if emb is None:
                continue
            pcs = emb["pcs"]
            embedding = emb["embedding"]
            n_cells = emb["n_cells"]

            valid_ks = [k for k in ks if k < n_cells]
            if not valid_ks:
                continue

            fig, axes = plt.subplots(
                1, len(valid_ks),
                figsize=(4.0 * len(valid_ks), 4.5),
                dpi=PLOT_PARAMS["dpi"],
            )
            if len(valid_ks) == 1:
                axes = np.array([axes])

            for col, k in enumerate(valid_ks):
                ax = axes[col]
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(top=False, right=False)

                km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labels = km.fit_predict(pcs)
                try:
                    s = silhouette_score(pcs, labels)
                except ValueError:
                    s = float("nan")

                for cid in range(k):
                    mask = labels == cid
                    ax.scatter(
                        embedding[mask, 0], embedding[mask, 1],
                        s=PLOT_PARAMS["scatter_size"] * 1.2,
                        color=cmap(cid % 10),
                        alpha=0.75,
                        edgecolors="none",
                        label=f"C{cid} (n={int(mask.sum())})",
                    )
                ax.set_xlabel("UMAP 1", fontsize=PLOT_PARAMS["axis_label_fontsize"])
                if col == 0:
                    ax.set_ylabel("UMAP 2", fontsize=PLOT_PARAMS["axis_label_fontsize"])
                ax.set_title(
                    f"k={k}  (silhouette={s:.3f})",
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"] - 1, loc="best")

            fig.suptitle(
                f"{exp_name} / {ch} — k-selection diagnostic (n={n_cells} cells)",
                fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                fontweight="bold",
                y=1.02,
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_trace_clustering_kselect"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


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
        figsize=(12, 5.5),
        dpi=PLOT_PARAMS["dpi"],
    )
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)

    ax = axes[0]
    ax.scatter(
        pcs[:, 0], pcs[:, 1],
        s=PLOT_PARAMS["scatter_size"] * 1.4,
        color=PLOT_PARAMS["scatter_color"],
        alpha=PLOT_PARAMS["scatter_alpha"],
        edgecolors="none",
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
        color=PLOT_PARAMS["scatter_color"],
        alpha=PLOT_PARAMS["scatter_alpha"],
        edgecolors="none",
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
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(
        fig_path(exp_name, save_name),
        dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
    )
    plt.close(fig)


def plot_pca_umap_uncolored(
    experiments, state, *, pool_channels=False, random_state=0,
):
    """PCA + UMAP scatters with all points one color (no cluster assignments)."""
    for exp_name, cfg in experiments.items():
        if pool_channels:
            channels = cfg["channels"]
            if not channels:
                continue
            _render_pca_umap_uncolored(
                state, exp_name, channels,
                f"pooled ({', '.join(channels)})",
                "pooled_pca_umap_uncolored",
                random_state=random_state,
            )
        else:
            for ch in cfg["channels"]:
                _render_pca_umap_uncolored(
                    state, exp_name, ch, ch,
                    f"{ch}_pca_umap_uncolored",
                    random_state=random_state,
                )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_trace_clustering(experiments, state)
    plot_trace_clustering_kselect(experiments, state)
    plot_trace_clustering(experiments, state, pool_channels=True)
    plot_pca_umap_uncolored(experiments, state)
    plot_pca_umap_uncolored(experiments, state, pool_channels=True)


if __name__ == "__main__":
    main()
