#!/usr/bin/env python3
"""Sliding-window pairwise Pearson + Spearman correlation traces.

Per (experiment, channel):
    * <ch>_sliding_corr.png

Note: this analysis is currently disabled in the source script's main() but is
kept here as a standalone option for parity. Computation is O(n_pairs × n_windows)
and can take minutes per channel.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS_SLIDING
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def plot_sliding_correlation(experiments, state):
    """Sliding-window Pearson + Spearman pairwise correlation traces."""
    pp = PLOT_PARAMS_SLIDING
    window_size = pp["window_size"]
    step = pp["step"]
    half_w = window_size // 2
    cutoff = pp["global_corr_cutoff"]
    sem_n = pp["sem_n"]

    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            lum_matrix = df[frame_cols].values
            n_cells, n_frames = lum_matrix.shape

            if n_cells < 2 or n_frames < window_size:
                print(
                    f"{exp_name} / {ch}: insufficient data "
                    f"(n_cells={n_cells}, n_frames={n_frames}) — skipping."
                )
                continue

            corr_full = pd.DataFrame(lum_matrix).T.corr(method="pearson").values
            i_idx, j_idx = np.triu_indices(n_cells, k=1)
            pw_pearson_full = corr_full[i_idx, j_idx]

            pair_mask = pw_pearson_full < cutoff
            n_pairs_all = len(pw_pearson_full)
            n_pairs_filtered = int(pair_mask.sum())
            if n_pairs_filtered < 2:
                print(
                    f"{exp_name} / {ch}: only {n_pairs_filtered} pairs below "
                    f"cutoff {cutoff} — skipping."
                )
                continue
            i_idx_f = i_idx[pair_mask]
            j_idx_f = j_idx[pair_mask]
            print(
                f"{exp_name} / {ch}: pairs {n_pairs_filtered:,} / "
                f"{n_pairs_all:,} ({100 * n_pairs_filtered / n_pairs_all:.1f}%) "
                f"below global Pearson {cutoff}"
            )

            centers = np.arange(half_w, n_frames - half_w, step)
            n_windows = len(centers)
            pearson_over_time = np.full((n_windows, n_pairs_filtered), np.nan)
            spearman_over_time = np.full((n_windows, n_pairs_filtered), np.nan)

            for wi, t in enumerate(
                tqdm(centers, desc=f"{exp_name} / {ch} sliding corr", leave=False)
            ):
                win = lum_matrix[:, t - half_w : t + half_w]
                for k, (i, j) in enumerate(zip(i_idx_f, j_idx_f)):
                    ti, tj = win[i], win[j]
                    if np.std(ti) < 1e-8 or np.std(tj) < 1e-8:
                        continue
                    pearson_over_time[wi, k], _ = pearsonr(ti, tj)
                    spearman_over_time[wi, k], _ = spearmanr(ti, tj)

            stim_frames = cfg["stim_frames"][ch]
            stim_min = frames_to_min(state, exp_name, ch, stim_frames) if stim_frames else []
            centers_min = frames_to_min(state, exp_name, ch, centers)
            rsp = state["real_setpoint_min"][exp_name].get(ch)

            fig, axes = plt.subplots(
                2, 1, figsize=pp["figsize"], dpi=pp["dpi"],
            )

            for ax, corr_mat, pair_color, mean_color, label in [
                (axes[0], pearson_over_time, pp["pearson_color"],
                 pp["mean_color_pearson"], "Pearson"),
                (axes[1], spearman_over_time, pp["spearman_color"],
                 pp["mean_color_spearman"], "Spearman"),
            ]:
                ax.spines[["top", "right"]].set_visible(False)
                ax.plot(
                    centers_min, corr_mat,
                    color=pair_color,
                    alpha=pp["line_alpha"],
                    lw=pp["line_lw"],
                )

                n_valid = np.sum(~np.isnan(corr_mat), axis=1)
                mean_corr = np.nanmean(corr_mat, axis=1)
                std_corr = np.nanstd(corr_mat, axis=1)
                sem_corr = np.where(
                    n_valid > 0, std_corr / np.sqrt(n_valid), np.nan
                )

                ax.fill_between(
                    centers_min,
                    mean_corr - sem_n * sem_corr,
                    mean_corr + sem_n * sem_corr,
                    color=mean_color, alpha=pp["sem_alpha"],
                    label=f"±{sem_n} SEM", zorder=4,
                )
                ax.plot(
                    centers_min, mean_corr,
                    color=mean_color, lw=pp["mean_lw"],
                    label=f"Mean {label}", zorder=5,
                )
                ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)

                for idx, p in enumerate(stim_min):
                    ax.axvline(
                        p,
                        color=pp["stim_color"],
                        linewidth=pp["stim_lw"],
                        alpha=1, zorder=0,
                        label="Stimulus" if idx == 0 else None,
                    )

                if rsp is not None:
                    ax.axvline(
                        rsp,
                        color="#000000",
                        linewidth=2.0, linestyle=":",
                        alpha=0.9, zorder=6,
                        label=f"Real setpoint ({rsp:.1f} min)",
                    )

                ax.set_xlabel("Time (min)", fontsize=pp["axis_label_fontsize"])
                ax.set_ylabel(
                    f"{label} correlation",
                    fontsize=pp["axis_label_fontsize"],
                )
                ax.set_title(
                    f"{label} (window = {window_size} frames, step = {step}, "
                    f"global Pearson < {cutoff})",
                    fontsize=pp["title_fontsize"],
                    fontweight=pp["title_fontweight"],
                )
                ax.legend(fontsize=pp["legend_fontsize"])

            fig.suptitle(
                f"{exp_name} / {ch} — sliding-window pairwise correlation "
                f"({n_pairs_filtered:,} of {n_pairs_all:,} pairs)",
                fontsize=pp["suptitle_fontsize"], fontweight="bold",
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_sliding_corr"),
                dpi=pp["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_sliding_correlation(experiments, state)


if __name__ == "__main__":
    main()
