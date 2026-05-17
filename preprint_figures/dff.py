#!/usr/bin/env python3
"""dF/F0 normalization figures.

Per (experiment, channel):
    * <ch>_dff.png                         — raw + dF/F0 stacked, all cells
    * <ch>_dff_responders.png              — raw + dF/F0 stacked, responders only
    * <ch>_dff_non_responders.png          — raw + dF/F0 stacked, non-responders only
Per experiment:
    * dff_mean_pooled_responders.png       — pooled mean ± SEM, responders only
    * dff_pooled_traces.png                — every dF/F0 trace pooled across
                                             channels + the mean, all cells
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import fig_path, save_fig
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_masks
from common.stim_helpers import (
    compute_f0_baseline,
    draw_stim_spans,
    stim_spans_min,
    stim_timing_aligned_across_channels,
)
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def plot_dff(experiments, state, *, responder_masks=None, select=None):
    """One figure per (experiment, channel): raw corrected + dF/F0 stacked.

    With ``responder_masks`` (the dict from ``compute_responder_masks``),
    ``select`` restricts which cells are plotted: ``"responders"`` keeps only
    responder cells, ``"non_responders"`` keeps only the rest, and the figure
    filename gains a matching ``_responders`` / ``_non_responders`` suffix.
    When ``select`` is None every cell is shown.
    """
    subset = select if responder_masks is not None else None
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

            n_all = mat.shape[0]
            if subset is not None:
                mask = responder_masks.get((exp_name, ch))
                mask = (
                    np.zeros(n_all, dtype=bool)
                    if mask is None
                    else np.asarray(mask, dtype=bool)
                )
                if subset == "non_responders":
                    mask = ~mask
                if not mask.any():
                    print(
                        f"  {exp_name} / {ch}: no {subset.replace('_', '-')} "
                        f"— skipping {subset.replace('_', '-')} dF/F₀ stack."
                    )
                    continue
                mat = mat[mask]
                dff_mat = dff_mat[mask]

            spans, stim_label = stim_spans_min(state, exp_name, ch, cfg)
            rsp = state["real_setpoint_min"][exp_name].get(ch)
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
                ax.legend(
                    fontsize=PLOT_PARAMS["legend_fontsize_large"], loc="upper right"
                )

            axes[-1].set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
            if subset == "responders":
                cell_str = f"{mat.shape[0]}/{n_all} responder cells"
            elif subset == "non_responders":
                cell_str = f"{mat.shape[0]}/{n_all} non-responder cells"
            else:
                cell_str = f"{mat.shape[0]} cells"
            fig.suptitle(
                f"{exp_name} / {ch} — {cell_str}, {len(stim_frames)} stims",
                fontsize=PLOT_PARAMS["title_fontsize"] + 1,
                fontweight="bold", y=1.01,
            )
            plt.tight_layout()
            name_suffix = {
                "responders": "_dff_responders",
                "non_responders": "_dff_non_responders",
                None: "_dff",
            }[subset]
            save_fig(
                fig, fig_path(exp_name, f"{ch}{name_suffix}"),
                dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
            )
            plt.close(fig)


def plot_dff_mean_pooled(experiments, state, *, responder_masks=None, only_responders=True):
    """Pool dF/F0 across channels and plot the mean ± SEM."""
    if only_responders and responder_masks is None:
        responder_masks = compute_responder_masks(experiments, state)

    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        if not channels:
            continue

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
                mask = responder_masks.get((exp_name, ch))
                if mask is None or not mask.any():
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
            color=PLOT_PARAMS["pooled_sem_color"], alpha=0.18, linewidth=0,
            label="±1 SEM",
        )
        ax.plot(
            frame_min, mean_trace,
            color=PLOT_PARAMS["pooled_mean_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=f"Pooled mean ({n_total} cells)",
            zorder=3,
        )

        if stim_timing_aligned_across_channels(cfg, state, exp_name):
            spans, stim_label = stim_spans_min(state, exp_name, ref_ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18,
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
        save_fig(
            fig,
            fig_path(
                exp_name,
                "dff_mean_pooled_responders" if only_responders else "dff_mean_pooled",
            ),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def plot_dff_pooled_traces(experiments, state):
    """Per experiment: every dF/F0 cell trace pooled across channels + the mean.

    Unlike ``plot_dff_mean_pooled`` (which shows mean ± SEM for responders
    only), this overlays every cell's normalized trace faintly with the pooled
    mean on top, and includes all cells (no responder filter) — the dF/F0
    panel of ``plot_dff``, but pooled across every channel of the experiment.
    """
    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        if not channels:
            continue

        per_ch_cols = {}
        for ch in channels:
            df = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            cols_ch = sorted(
                [c for c in df.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            per_ch_cols[ch] = (df, cols_ch)
        if any(not cols_ch for _, cols_ch in per_ch_cols.values()):
            continue
        n_common = min(len(cols_ch) for _, cols_ch in per_ch_cols.values())
        if any(len(cols_ch) != n_common for _, cols_ch in per_ch_cols.values()):
            print(
                f"  {exp_name}: pooled dF/F₀ traces — channel frame counts "
                f"{[len(cols_ch) for _, cols_ch in per_ch_cols.values()]} "
                f"differ; truncating to common length {n_common}."
            )

        ref_ch = channels[0]
        ref_frame_cols = per_ch_cols[ref_ch][1][:n_common]
        ref_frame_nums = np.array(
            [int(str(c).lstrip("f")) for c in ref_frame_cols]
        )
        frame_min = frames_to_min(state, exp_name, ref_ch, ref_frame_nums)

        pooled_rows = []
        per_channel_counts = []
        for ch in channels:
            df, frame_cols = per_ch_cols[ch]
            mat = df[frame_cols[:n_common]].values
            F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            pooled_rows.append((mat - F0) / F0_safe)
            per_channel_counts.append((ch, int(mat.shape[0])))

        pooled = np.vstack(pooled_rows)
        n_total = pooled.shape[0]
        mean_trace = np.nanmean(pooled, axis=0)

        fig, ax = plt.subplots(
            figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        for row in pooled:
            ax.plot(
                frame_min, row,
                color=PLOT_PARAMS["cell_color"],
                alpha=PLOT_PARAMS["cell_alpha"],
                linewidth=PLOT_PARAMS["cell_lw"], zorder=1,
            )
        ax.plot(
            frame_min, mean_trace,
            color=PLOT_PARAMS["pooled_mean_color"],
            linewidth=PLOT_PARAMS["mean_lw"],
            label=f"Pooled mean ({n_total} cells)", zorder=3,
        )
        if stim_timing_aligned_across_channels(cfg, state, exp_name):
            spans, stim_label = stim_spans_min(state, exp_name, ref_ch, cfg)
            draw_stim_spans(
                ax, spans, stim_label, PLOT_PARAMS["stim_color"], alpha=0.18,
            )
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
        ax.set_xlabel("Time (min)", fontsize=PLOT_PARAMS["axis_label_fontsize"])
        ax.set_ylabel(
            "dF/F₀  (pooled across channels)",
            fontsize=PLOT_PARAMS["axis_label_fontsize"],
        )
        per_ch_str = ", ".join(f"{ch}: {n}" for ch, n in per_channel_counts)
        ax.set_title(
            f"{exp_name} — pooled dF/F₀ traces, all cells\n[{per_ch_str}]",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
        plt.tight_layout()
        save_fig(
            fig, fig_path(exp_name, "dff_pooled_traces"),
            dpi=PLOT_PARAMS["dpi"], bbox_inches="tight",
        )
        plt.close(fig)


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_dff(experiments, state)
    responder_masks = compute_responder_masks(experiments, state)
    plot_dff(
        experiments, state, responder_masks=responder_masks,
        select="responders",
    )
    plot_dff(
        experiments, state, responder_masks=responder_masks,
        select="non_responders",
    )
    plot_dff_mean_pooled(
        experiments, state, responder_masks=responder_masks,
        only_responders=True,
    )
    plot_dff_pooled_traces(experiments, state)


if __name__ == "__main__":
    main()
