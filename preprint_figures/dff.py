#!/usr/bin/env python3
"""dF/F0 normalization figures.

Per (experiment, channel):
    * <ch>_dff.png                         — raw + dF/F0 stacked
Per experiment:
    * dff_mean_combined.png                — mean dF/F0 per channel overlaid
    * dff_mean_pooled_responders.png       — pooled mean ± SEM, responders only
    * <ch>_dff_response_breakdown.png      — c2c12 only, per-channel histogram
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.cluster_labels import align_labels_to_cells, load_cluster_labels
from common.config import PEAK_OFFSET
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_thresholds
from common.stim_helpers import (
    compute_f0_baseline,
    draw_stim_spans,
    per_cell_response_delta,
    stim_spans_min,
    stim_timing_aligned_across_channels,
)
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def plot_dff(experiments, state):
    """One figure per (experiment, channel): raw corrected + dF/F0 stacked."""
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            F0, baseline_cols, first_stim = compute_f0_baseline(
                state, exp_name, ch, cfg
            )
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            rsp = state["real_setpoint_min"][exp_name].get(ch)
            baseline_lo_min = frames_to_min(state, exp_name, ch, [0])[0]
            baseline_hi_min = frames_to_min(
                state, exp_name, ch, [max(first_stim - 1, 0)]
            )[0]
            baseline_label = (
                f"F₀ baseline (frames 0–{first_stim - 1})"
                if first_stim > 1
                else "F₀ baseline (frame 0)"
            )
            f0_title_suffix = (
                f"F₀ = mean of baseline (frames 0–{first_stim - 1})"
                if first_stim > 1
                else "F₀ = frame 0"
            )

            fig, axes = plt.subplots(
                2, 1,
                figsize=PLOT_PARAMS["figsize_wide"],
                dpi=PLOT_PARAMS["dpi"], sharex=True,
            )

            panels = [
                (mat, "Corrected luminosity", "Corrected luminosity"),
                (dff_mat, "dF/F₀", f"dF/F₀  ({f0_title_suffix})"),
            ]

            for ax, (data, ylabel, title_suffix) in zip(axes, panels):
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(top=False, right=False)

                mean_trace = np.nanmean(data, axis=0)
                for row in data:
                    ax.plot(
                        frame_min, row,
                        color=PLOT_PARAMS["cell_color"],
                        alpha=PLOT_PARAMS["cell_alpha"],
                        linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
                    )
                ax.plot(
                    frame_min, mean_trace,
                    color=PLOT_PARAMS["mean_color"],
                    linewidth=PLOT_PARAMS["mean_lw"], zorder=3,
                    label="Mean",
                )

                draw_stim_spans(
                    ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
                )

                ax.axvspan(
                    baseline_lo_min, baseline_hi_min,
                    color=PLOT_PARAMS["f0_color"], alpha=0.10, zorder=0,
                    label=baseline_label,
                )

                if rsp is not None:
                    ax.axvline(
                        rsp,
                        color="#000000",
                        linewidth=2.0, linestyle=":",
                        alpha=0.9, zorder=5,
                        label=f"Real setpoint ({rsp:.1f} min)",
                    )

                ax.set_ylabel(ylabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
                ax.set_title(
                    title_suffix,
                    fontsize=PLOT_PARAMS["title_fontsize"],
                    fontweight=PLOT_PARAMS["title_fontweight"],
                )
                ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="upper right")

            axes[-1].set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            fig.suptitle(
                f"{exp_name} / {ch} — {mat.shape[0]} cells, {len(stim_frames)} stims",
                fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                fontweight="bold", y=1.01,
            )
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_dff"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


def plot_dff_mean_combined(experiments, state):
    """One figure per experiment: mean dF/F0 trace from every channel overlaid."""
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]

        fig, ax = plt.subplots(
            figsize=PLOT_PARAMS["figsize"],
            dpi=PLOT_PARAMS["dpi"],
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)

        cmap = plt.get_cmap("tab10")
        any_drawn = False
        ref_first_stim = None
        for col, ch in enumerate(channels):
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            if not frame_cols:
                continue
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            F0, _, first_stim = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe
            mean_trace = np.nanmean(dff_mat, axis=0)
            if ref_first_stim is None:
                ref_first_stim = first_stim

            ax.plot(
                frame_min, mean_trace,
                color=cmap(col % 10),
                linewidth=PLOT_PARAMS["mean_lw"],
                label=f"{ch} ({mat.shape[0]} cells)",
                zorder=3,
            )
            any_drawn = True

        if not any_drawn:
            plt.close(fig)
            continue

        ref_ch = channels[0]
        if stim_timing_aligned_across_channels(cfg, state, exp_name):
            spans, stim_label = stim_spans_min(state, exp_name, ref_ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
            )

        if ref_first_stim is not None:
            base_lo = frames_to_min(state, exp_name, ref_ch, [0])[0]
            base_hi = frames_to_min(
                state, exp_name, ref_ch, [max(ref_first_stim - 1, 0)]
            )[0]
            ax.axvspan(
                base_lo, base_hi,
                color=PLOT_PARAMS["f0_color"], alpha=0.10, zorder=0,
                label=(
                    f"F₀ baseline (frames 0–{ref_first_stim - 1})"
                    if ref_first_stim > 1
                    else "F₀ baseline (frame 0)"
                ),
            )

        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_ylabel(
            "Mean dF/F₀  (F₀ = baseline-window mean)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        ax.set_title(
            f"{exp_name} — mean dF/F₀ per channel",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
        plt.tight_layout()
        fig.savefig(
            fig_path(exp_name, "dff_mean_combined"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def plot_dff_mean_pooled(experiments, state, *, thresholds=None, only_responders=True):
    """Pool dF/F0 across channels and plot the mean ± SEM."""
    if only_responders and thresholds is None:
        thresholds = compute_responder_thresholds(experiments, state)

    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        if not channels:
            continue
        direction = cfg.get("response_direction", "increase")
        sign = -1.0 if direction == "decrease" else 1.0

        ref_ch = channels[0]
        df_ref = lum_dict_to_df(state["corrected_lum"][exp_name][ref_ch]).set_index("CellID")
        ref_frame_cols = sorted(
            [c for c in df_ref.columns if str(c).startswith("f")],
            key=lambda c: int(str(c).lstrip("f")),
        )
        if not ref_frame_cols:
            continue
        per_ch_cols = {}
        for ch in channels:
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            cols_ch = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            per_ch_cols[ch] = (df, cols_ch)
        n_common = min(len(cols_ch) for _, cols_ch in per_ch_cols.values())
        if any(len(cols_ch) != n_common for _, cols_ch in per_ch_cols.values()):
            print(
                f"  {exp_name}: pooled mean dF/F₀ — channel frame counts "
                f"{[len(cols_ch) for _, cols_ch in per_ch_cols.values()]} "
                f"differ; truncating to common length {n_common}."
            )
        ref_frame_nums = np.array(
            [int(str(c).lstrip("f")) for c in ref_frame_cols[:n_common]]
        )
        frame_min = frames_to_min(state, exp_name, ref_ch, ref_frame_nums)

        pooled_rows = []
        per_channel_counts = []
        for ch in channels:
            df, frame_cols = per_ch_cols[ch]
            frame_cols = frame_cols[:n_common]
            mat = df[frame_cols].values
            F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            if only_responders:
                stim_frames = cfg["stim_frames"][ch]
                frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
                frame_to_col = {f: i for i, f in enumerate(frame_nums)}
                window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
                per_stim = []
                for p in stim_frames:
                    if p not in frame_to_col:
                        continue
                    per_stim.append(
                        per_cell_response_delta(
                            dff_mat, frame_to_col[p], direction, window
                        )
                    )
                if not per_stim:
                    continue
                stacked = np.vstack(per_stim)
                if direction == "decrease":
                    per_cell_peak = np.nanmin(stacked, axis=0)
                else:
                    per_cell_peak = np.nanmax(stacked, axis=0)
                t = float(thresholds.get((exp_name, ch), 0.10))
                signed_t = sign * t
                if direction == "decrease":
                    mask = per_cell_peak <= signed_t
                else:
                    mask = per_cell_peak >= signed_t
                mask = mask & ~np.isnan(per_cell_peak)
                if not mask.any():
                    continue
                pooled_rows.append(dff_mat[mask])
                per_channel_counts.append((ch, int(mask.sum()), int(mat.shape[0])))
            else:
                pooled_rows.append(dff_mat)
                per_channel_counts.append((ch, int(mat.shape[0]), int(mat.shape[0])))

        if not pooled_rows:
            print(f"{exp_name}: no responders pooled — skipping pooled mean dF/F₀.")
            continue
        pooled = np.vstack(pooled_rows)
        n_total = pooled.shape[0]
        mean_trace = np.nanmean(pooled, axis=0)
        n_per_col = np.sum(~np.isnan(pooled), axis=0).astype(float)
        sem_trace = np.nanstd(pooled, axis=0) / np.sqrt(np.maximum(n_per_col, 1))

        fig, ax = plt.subplots(
            figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.fill_between(
            frame_min, mean_trace - sem_trace, mean_trace + sem_trace,
            color=PLOT_PARAMS["mean_color"], alpha=0.18, linewidth=0,
            label="±1 SEM",
        )
        ax.plot(
            frame_min, mean_trace,
            color=PLOT_PARAMS["mean_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=f"Pooled mean ({n_total} cells)",
            zorder=3,
        )

        if stim_timing_aligned_across_channels(cfg, state, exp_name):
            spans, stim_label = stim_spans_min(state, exp_name, ref_ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18,
            )
        _, _, ref_first_stim = compute_f0_baseline(state, exp_name, ref_ch, cfg)
        base_lo = frames_to_min(state, exp_name, ref_ch, [0])[0]
        base_hi = frames_to_min(
            state, exp_name, ref_ch, [max(ref_first_stim - 1, 0)]
        )[0]
        ax.axvspan(
            base_lo, base_hi,
            color=PLOT_PARAMS["f0_color"], alpha=0.10, zorder=0,
            label=(
                f"F₀ baseline (frames 0–{ref_first_stim - 1})"
                if ref_first_stim > 1
                else "F₀ baseline (frame 0)"
            ),
        )
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_ylabel(
            "Mean dF/F₀  (pooled across channels)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        suffix = " — responders only" if only_responders else ""
        per_ch_str = ", ".join(
            f"{ch}: {n}/{tot}" for ch, n, tot in per_channel_counts
        )
        ax.set_title(
            f"{exp_name} — pooled mean dF/F₀{suffix}\n[{per_ch_str}]",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
        plt.tight_layout()
        fig.savefig(
            fig_path(
                exp_name,
                "dff_mean_pooled_responders" if only_responders else "dff_mean_pooled",
            ),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def plot_dff_response_diagnostic(experiments, state, only_experiments=("c2c12_dmso_09APR26",)):
    """Per-channel diagnostic: responder fraction vs responder magnitude."""
    for exp_name, cfg in experiments.items():
        if exp_name not in only_experiments:
            continue
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        cutoff_mags = (0.02, 0.05, 0.10)
        sign = -1.0 if direction == "decrease" else 1.0
        cutoffs = tuple(sign * c for c in cutoff_mags)

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            F0, _, first_stim = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            frame_to_col = {f: i for i, f in enumerate(frame_nums.tolist())}
            n_cells = dff_mat.shape[0]
            per_stim = []
            for p in stim_frames:
                if p not in frame_to_col:
                    continue
                col = frame_to_col[p]
                deltas = per_cell_response_delta(dff_mat, col, direction, window)
                per_stim.append(deltas)
            if per_stim:
                stacked = np.vstack(per_stim)
                if direction == "decrease":
                    per_cell_peak = np.nanmin(stacked, axis=0)
                else:
                    per_cell_peak = np.nanmax(stacked, axis=0)
            else:
                per_cell_peak = np.full(n_cells, np.nan)
            per_cell_peak = per_cell_peak[~np.isnan(per_cell_peak)]

            mean_trace = np.nanmean(dff_mat, axis=0)
            median_trace = np.nanmedian(dff_mat, axis=0)
            p75_trace = np.nanpercentile(dff_mat, 75, axis=0)
            p90_trace = np.nanpercentile(dff_mat, 90, axis=0)

            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            baseline_lo_min = frames_to_min(state, exp_name, ch, [0])[0]
            baseline_hi_min = frames_to_min(
                state, exp_name, ch, [max(first_stim - 1, 0)]
            )[0]
            baseline_label = (
                f"F₀ baseline (frames 0–{first_stim - 1})"
                if first_stim > 1
                else "F₀ baseline (frame 0)"
            )

            fig, axes = plt.subplots(
                2, 1,
                figsize=(14, 9),
                dpi=PLOT_PARAMS["dpi"],
                gridspec_kw={"height_ratios": [3, 2]},
            )

            ax_top = axes[0]
            ax_top.spines[["top", "right"]].set_visible(False)
            ax_top.tick_params(top=False, right=False)
            for row in dff_mat:
                ax_top.plot(
                    frame_min, row,
                    color=PLOT_PARAMS["cell_color"],
                    alpha=PLOT_PARAMS["cell_alpha"],
                    linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
                )
            for trace, color, label, lw in [
                (mean_trace,   "#1a1a1a", "Mean",   2.0),
                (median_trace, "#1aa821", "Median", 2.0),
                (p75_trace,    "#e67e22", "75th percentile", 1.6),
                (p90_trace,    "#c0392b", "90th percentile", 1.6),
            ]:
                ax_top.plot(
                    frame_min, trace,
                    color=color, linewidth=lw,
                    label=label, zorder=4,
                )
            draw_stim_spans(
                ax_top, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18
            )
            ax_top.axvspan(
                baseline_lo_min, baseline_hi_min,
                color=PLOT_PARAMS["f0_color"], alpha=0.10, zorder=0,
                label=baseline_label,
            )
            ax_top.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6, zorder=2)
            ax_top.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax_top.set_ylabel("dF/F₀", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax_top.set_title(
                f"{exp_name} / {ch} — dF/F₀ summary curves "
                f"(mean / median / 75th / 90th)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax_top.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="upper right")

            ax_bot = axes[1]
            ax_bot.spines[["top", "right"]].set_visible(False)
            ax_bot.tick_params(top=False, right=False)
            if len(per_cell_peak) > 0:
                ax_bot.hist(
                    per_cell_peak,
                    bins=60,
                    color=PLOT_PARAMS["violin_face"],
                    edgecolor=PLOT_PARAMS["violin_edge"],
                    linewidth=0.6,
                    zorder=2,
                )
            cutoff_colors = ["#1aa821", "#e67e22", "#c0392b"]
            cmp_op = (lambda v, c: v <= c) if direction == "decrease" else (lambda v, c: v >= c)
            cmp_str = "≤" if direction == "decrease" else "≥"
            for cut, ccolor in zip(cutoffs, cutoff_colors):
                frac = float(cmp_op(per_cell_peak, cut).mean()) if len(per_cell_peak) else 0.0
                ax_bot.axvline(
                    cut, color=ccolor, linewidth=1.5, linestyle="--",
                    label=f"{cmp_str} {cut:+g}: {frac * 100:.1f}% of cells",
                    zorder=3,
                )
            extremum_label = "max" if direction == "increase" else "min"
            ax_bot.set_xlabel(
                f"Per-cell peak Δ dF/F₀  (largest-magnitude {extremum_label}−baseline across stims)",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            ax_bot.set_ylabel("Cell count", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax_bot.set_title(
                f"Per-cell peak Δ dF/F₀ distribution  (n = {len(per_cell_peak)} of {n_cells} cells)",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax_bot.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="upper right")

            mean_peak = float(np.nanmean(per_cell_peak)) if len(per_cell_peak) else float("nan")
            med_peak = float(np.nanmedian(per_cell_peak)) if len(per_cell_peak) else float("nan")
            fracs = [
                float(cmp_op(per_cell_peak, c).mean()) * 100 if len(per_cell_peak) else 0.0
                for c in cutoffs
            ]
            print(
                f"{exp_name} / {ch}: peak Δ dF/F0 ({direction}) — "
                f"mean={mean_peak:+.4f}, median={med_peak:+.4f}, "
                f"frac {cmp_str} {cutoffs[0]:+g} = {fracs[0]:.1f}%, "
                f"frac {cmp_str} {cutoffs[1]:+g} = {fracs[1]:.1f}%, "
                f"frac {cmp_str} {cutoffs[2]:+g} = {fracs[2]:.1f}%"
            )

            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_dff_response_breakdown"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


def plot_dff_mean_per_cluster(experiments, state):
    """Per (exp, ch): mean ± SEM dF/F0 trace per PCA cluster.

    Reads cluster labels written by ``clustering.py``. Skips silently when no
    cluster cache exists (e.g. before clustering has run).
    """
    cmap = plt.get_cmap("tab10")
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            blob = load_cluster_labels(exp_name, ch)
            if blob is None:
                continue
            df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
            frame_cols = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            if not frame_cols:
                continue
            frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
            frame_min = frames_to_min(state, exp_name, ch, frame_nums)
            mat = df[frame_cols].values

            F0, _, first_stim = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            target_ids = [(ch, cid) for cid in df.index]
            cluster_labels = align_labels_to_cells(blob, target_ids)
            best_k = int(blob["best_k"])

            fig, ax = plt.subplots(
                figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(top=False, right=False)
            for cid in range(best_k):
                mask = cluster_labels == cid
                n_in = int(mask.sum())
                if n_in < 3:
                    continue
                cluster_dff = dff_mat[mask]
                mean_trace = np.nanmean(cluster_dff, axis=0)
                n_per_col = np.sum(~np.isnan(cluster_dff), axis=0).astype(float)
                sem_trace = np.nanstd(cluster_dff, axis=0) / np.sqrt(
                    np.maximum(n_per_col, 1)
                )
                color = cmap(cid % 10)
                ax.fill_between(
                    frame_min, mean_trace - sem_trace, mean_trace + sem_trace,
                    color=color, alpha=0.18, linewidth=0,
                )
                ax.plot(
                    frame_min, mean_trace,
                    color=color,
                    linewidth=PLOT_PARAMS["mean_lw"],
                    label=f"Cluster {cid} (n={n_in})",
                )

            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18,
            )
            base_lo = frames_to_min(state, exp_name, ch, [0])[0]
            base_hi = frames_to_min(
                state, exp_name, ch, [max(first_stim - 1, 0)]
            )[0]
            ax.axvspan(
                base_lo, base_hi,
                color=PLOT_PARAMS["f0_color"], alpha=0.10, zorder=0,
                label=(
                    f"F₀ baseline (frames 0–{first_stim - 1})"
                    if first_stim > 1 else "F₀ baseline (frame 0)"
                ),
            )
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
            ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            ax.set_ylabel(
                "Mean dF/F₀ ± 1 SEM",
                fontsize=PLOT_PARAMS["axis_label_fontsize"],
            )
            ax.set_title(
                f"{exp_name} / {ch} — mean dF/F₀ per cluster (k={best_k})",
                fontsize=PLOT_PARAMS["title_fontsize"],
                fontweight=PLOT_PARAMS["title_fontweight"],
            )
            ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
            plt.tight_layout()
            fig.savefig(
                fig_path(exp_name, f"{ch}_dff_mean_per_cluster"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_dff(experiments, state)
    plot_dff_mean_combined(experiments, state)
    plot_dff_response_diagnostic(experiments, state)
    thresholds = compute_responder_thresholds(experiments, state, alpha=0.01)
    plot_dff_mean_pooled(
        experiments, state, thresholds=thresholds, only_responders=True,
    )
    plot_dff_mean_per_cluster(experiments, state)


if __name__ == "__main__":
    main()
