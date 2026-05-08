#!/usr/bin/env python3
"""Per-stimulus response violins (asymmetric half-violin + notched box).

Per (experiment, channel):
    * <ch>_response_violin.png             — height (Δ luminosity)
    * <ch>_response_violin_width.png       — width (response duration in min)
    * <ch>_response_violin_responders.png  — responders only
Per experiment:
    * pooled_response_violin_height.png
    * pooled_response_violin_width.png

The absolute peak-luminosity violin (plot_per_stimulus_peak_violins) is kept
for completeness but disabled per reviewer request — only peak−baseline deltas
are emitted.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import PEAK_OFFSET
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.responders import compute_responder_thresholds
from common.stim_helpers import (
    compute_f0_baseline,
    compute_stim_caps,
    per_cell_response_delta,
)
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


_VIOLIN_BOX_OFFSET = 0.18
_VIOLIN_BOX_WIDTH = 0.18
_VIOLIN_SCATTER_JITTER = 0.045


def _draw_half_violin_with_box(ax, violin_data, x_label, y_label, title, save_path,
                                x_axis_label="Peak frame"):
    """Render the asymmetric violin/box composite for one figure and save it."""
    n_cat = len(violin_data)
    non_empty_idx = [i for i, v in enumerate(violin_data) if len(v) > 0]
    if not non_empty_idx:
        return False

    fig, ax_ = plt.subplots(
        figsize=(max(6, n_cat * 1.2), 10),
        dpi=PLOT_PARAMS["dpi"],
    )
    ax_.spines[["top", "right"]].set_visible(False)
    ax_.tick_params(top=False, right=False)

    box_x = [i - _VIOLIN_BOX_OFFSET for i in non_empty_idx]
    box_data = [violin_data[i] for i in non_empty_idx]

    rng = np.random.default_rng(42)
    for col_idx, bx in zip(non_empty_idx, box_x):
        vd = violin_data[col_idx]
        xs = bx + rng.uniform(
            -_VIOLIN_SCATTER_JITTER, _VIOLIN_SCATTER_JITTER, size=len(vd)
        )
        ax_.scatter(
            xs, vd,
            color=PLOT_PARAMS["scatter_color"],
            alpha=PLOT_PARAMS["scatter_alpha"],
            s=PLOT_PARAMS["scatter_size"],
            zorder=2, linewidths=0,
        )

    bp = ax_.boxplot(
        box_data,
        positions=box_x,
        widths=_VIOLIN_BOX_WIDTH,
        notch=True,
        bootstrap=None,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(
            color=PLOT_PARAMS["median_color"], linewidth=2.0
        ),
        boxprops=dict(
            facecolor=PLOT_PARAMS["violin_face"],
            edgecolor=PLOT_PARAMS["violin_edge"],
            linewidth=1.2,
        ),
        whiskerprops=dict(color=PLOT_PARAMS["violin_edge"], linewidth=1.0),
        capprops=dict(color=PLOT_PARAMS["violin_edge"], linewidth=1.0),
        zorder=3,
    )
    median_handles = bp.get("medians", [])
    for i, handle in enumerate(median_handles):
        handle.set_label("Median" if i == 0 else None)

    vp = ax_.violinplot(
        box_data,
        positions=non_empty_idx,
        showmedians=False,
        showextrema=False,
        showmeans=False,
    )
    for body, pos in zip(vp["bodies"], non_empty_idx):
        for path in body.get_paths():
            verts = path.vertices
            verts[verts[:, 0] < pos, 0] = pos
        body.set_facecolor(PLOT_PARAMS["violin_face"])
        body.set_edgecolor(PLOT_PARAMS["violin_edge"])
        body.set_alpha(0.85)
        body.set_zorder(3)

    means = [float(np.mean(v)) for v in box_data]
    ax_.scatter(
        non_empty_idx, means,
        marker="_",
        color=PLOT_PARAMS["mean_marker_color"],
        s=200, linewidths=2.5,
        zorder=6, label="Mean",
    )

    ax_.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="center right")

    ax_.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax_.set_xticks(range(n_cat))
    ax_.set_xticklabels([str(lbl) for lbl in x_label], fontsize=9)
    ax_.set_xlabel(x_axis_label, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax_.set_ylabel(y_label, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    plt.tight_layout()
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)
    return True


def plot_per_stimulus_peak_violins(experiments, state):
    """Violin of corrected luminosity at the peak frame for every stimulus.

    DISABLED per reviewer request — only peak−baseline deltas are emitted by
    the pipeline. Kept here for completeness; not called from main().
    """
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            if not stim_frames:
                print(f"{exp_name} / {ch}: no stim_frames — skipping peak violin.")
                continue

            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            peak_cols = [f"f{p + PEAK_OFFSET}" for p in stim_frames]
            available_cols = [c for c in peak_cols if c in df_indexed.columns]
            if not available_cols:
                print(f"{exp_name} / {ch}: no peak columns available, skipping.")
                continue
            complete_df = df_indexed[available_cols].dropna()
            print(
                f"{exp_name} / {ch}: peak violin — {len(complete_df)} cells "
                f"complete across {len(available_cols)} peaks."
            )

            violin_data = [
                df_indexed[f"f{p + PEAK_OFFSET}"].dropna().values
                if f"f{p + PEAK_OFFSET}" in df_indexed.columns
                else np.array([])
                for p in stim_frames
            ]
            peak_min = frames_to_min(
                state, exp_name, ch, [p + PEAK_OFFSET for p in stim_frames]
            )
            x_labels = [f"{m:.1f}" for m in peak_min]
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label="Corrected luminosity at peak",
                title=(
                    f"{exp_name} / {ch} — peak luminosity at each stimulus "
                    f"({len(complete_df)} complete cells)"
                ),
                save_path=fig_path(exp_name, f"{ch}_peak_value_violin"),
                x_axis_label="Peak time (min)",
            )
            if not ok:
                print(f"{exp_name} / {ch}: no non-empty peak data — skipped.")


def _per_channel_stim_response_arrays(state, exp_name, ch, cfg, *, metric):
    """Return ``(violin_data, base_min, n_total)`` for one (exp, channel)."""
    direction = cfg.get("response_direction", "increase")
    window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))

    stim_frames = cfg["stim_frames"][ch]
    df_indexed = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df_indexed.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
    mat = df_indexed[frame_cols].values
    n_cells, n_cols = mat.shape
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}

    valid_stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if metric == "width" and valid_stim_cols:
        if len(valid_stim_cols) >= 2:
            uniform_cap = int(np.min(np.diff(valid_stim_cols)))
        else:
            uniform_cap = max(0, n_cols - 1 - valid_stim_cols[0])
        caps = compute_stim_caps(
            valid_stim_cols, n_cols, uniform_cap_cols=uniform_cap,
        )
    else:
        caps = []
    cap_for_col = dict(zip(valid_stim_cols, caps))

    def _f2m(frames):
        return frames_to_min(state, exp_name, ch, frames)

    violin_data = []
    for p in stim_frames:
        if p not in frame_to_col:
            violin_data.append(np.array([]))
            continue
        col = frame_to_col[p]
        if metric == "width":
            _, widths = per_cell_response_delta(
                mat, col, direction, window,
                return_width=True,
                cap_col=cap_for_col[col],
                frame_to_min_fn=_f2m,
            )
            vals = widths[~np.isnan(widths)]
        else:
            deltas = per_cell_response_delta(mat, col, direction, window)
            vals = deltas[~np.isnan(deltas)]
        violin_data.append(vals)

    base_min = list(frames_to_min(state, exp_name, ch, stim_frames)) if stim_frames else []
    return violin_data, base_min, n_cells


def plot_per_stimulus_response_violins(
    experiments, state, *, metric="height", pool_channels=False,
):
    """Violin of per-cell response per stim."""
    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        extremum_label = "max" if direction == "increase" else "min"
        window_str = f"stim+{window[0]}…stim+{window[1] - 1} frames"
        if metric == "width":
            y_label = "Response width (min)"
            width_cap_note = (
                " — width search capped at min inter-stim interval"
            )
        else:
            y_label = f"Δ luminosity  ({extremum_label} − baseline)"
            width_cap_note = ""

        if pool_channels:
            channels = cfg["channels"]
            stim_counts = [len(cfg["stim_frames"][ch]) for ch in channels]
            if not channels or len(set(stim_counts)) != 1 or stim_counts[0] == 0:
                print(
                    f"{exp_name}: pooled response violin ({metric}) — channels "
                    f"have differing stim counts {stim_counts} or no stims; skipping."
                )
                continue
            n_stims = stim_counts[0]
            pooled = [[] for _ in range(n_stims)]
            ref_base_min = None
            total_cells = 0
            for ch in channels:
                vd, base_min, n_cells = _per_channel_stim_response_arrays(
                    state, exp_name, ch, cfg, metric=metric,
                )
                if ref_base_min is None:
                    ref_base_min = base_min
                total_cells += n_cells
                for i, arr in enumerate(vd):
                    if arr.size:
                        pooled[i].append(arr)
            violin_data = [
                np.concatenate(p) if p else np.array([]) for p in pooled
            ]
            x_labels = [f"{bm:.1f}" for bm in (ref_base_min or [])]
            n_complete = sum(len(v) > 0 for v in violin_data)
            print(
                f"{exp_name}: pooled response violin ({metric}) — "
                f"{n_complete}/{n_stims} stims have data, {total_cells} cells "
                f"across {len(channels)} channels."
            )
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label=y_label,
                title=(
                    f"{exp_name} — pooled per-stimulus {metric} "
                    f"({extremum_label} over {window_str} − baseline; "
                    f"{total_cells} cells, {len(channels)} channels)"
                    f"{width_cap_note}"
                ),
                save_path=fig_path(
                    exp_name, f"pooled_response_violin_{metric}"
                ),
                x_axis_label="Stimulus onset (min)",
            )
            if not ok:
                print(
                    f"{exp_name}: pooled response violin ({metric}) — "
                    "no non-empty data; skipped."
                )
            continue

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            if not stim_frames:
                print(
                    f"{exp_name} / {ch}: no stim_frames — "
                    f"skipping response violin ({metric})."
                )
                continue

            violin_data, base_min, _ = _per_channel_stim_response_arrays(
                state, exp_name, ch, cfg, metric=metric,
            )
            n_complete = sum(len(v) > 0 for v in violin_data)
            print(
                f"{exp_name} / {ch}: response violin ({metric}) — "
                f"{n_complete}/{len(stim_frames)} stimuli have data."
            )
            x_labels = [f"{bm:.1f}" for bm in base_min]
            metric_suffix = "" if metric == "height" else f"_{metric}"
            title = (
                f"{exp_name} / {ch} — per-stimulus {metric} "
                f"({extremum_label} over {window_str} − baseline)"
                f"{width_cap_note}"
            )
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label=y_label,
                title=title,
                save_path=fig_path(
                    exp_name, f"{ch}_response_violin{metric_suffix}"
                ),
                x_axis_label="Stimulus onset (min)",
            )
            if not ok:
                print(
                    f"{exp_name} / {ch}: no non-empty response data ({metric}) — "
                    "skipped."
                )


def plot_per_stimulus_response_violins_responders(experiments, state, thresholds=None):
    """Same as the response violin but restricted to responders."""
    if thresholds is None:
        thresholds = compute_responder_thresholds(experiments, state)

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        sign = -1.0 if direction == "decrease" else 1.0

        for ch in cfg["channels"]:
            stim_frames = cfg["stim_frames"][ch]
            if not stim_frames:
                print(
                    f"{exp_name} / {ch}: no stim_frames — skipping responder violin."
                )
                continue

            dff_threshold = float(thresholds.get((exp_name, ch), 0.10))
            signed_threshold = sign * dff_threshold

            df_indexed = lum_dict_to_df(
                state["corrected_lum"][exp_name][ch]
            ).set_index("CellID")
            frame_cols = sorted(
                [c for c in df_indexed.columns if str(c).startswith("f")],
                key=lambda c: int(str(c).lstrip("f")),
            )
            frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
            mat = df_indexed[frame_cols].values
            frame_to_col = {f: i for i, f in enumerate(frame_nums)}

            F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            F0_safe = np.where(F0 == 0, np.nan, F0)
            dff_mat = (mat - F0) / F0_safe

            per_stim_dff = []
            per_stim_lum = []
            stim_cols_used = []
            for p in stim_frames:
                if p not in frame_to_col:
                    continue
                col = frame_to_col[p]
                stim_cols_used.append((p, col))
                per_stim_dff.append(
                    per_cell_response_delta(dff_mat, col, direction, window)
                )
                per_stim_lum.append(
                    per_cell_response_delta(mat, col, direction, window)
                )
            if not per_stim_dff:
                print(f"{exp_name} / {ch}: no usable stims — skipping responder violin.")
                continue

            stacked_dff = np.vstack(per_stim_dff)
            stacked_lum = np.vstack(per_stim_lum)
            if direction == "decrease":
                per_cell_peak_dff = np.nanmin(stacked_dff, axis=0)
                responder_mask = per_cell_peak_dff <= signed_threshold
            else:
                per_cell_peak_dff = np.nanmax(stacked_dff, axis=0)
                responder_mask = per_cell_peak_dff >= signed_threshold
            n_total = mat.shape[0]
            n_resp = int(np.sum(responder_mask & ~np.isnan(per_cell_peak_dff)))

            if n_resp == 0:
                print(
                    f"{exp_name} / {ch}: 0 responders at |dF/F0| ≥ {dff_threshold} — "
                    "skipping responder violin."
                )
                continue

            violin_data = []
            for row in stacked_lum:
                vals = row[responder_mask]
                vals = vals[~np.isnan(vals)]
                violin_data.append(vals)

            base_min = frames_to_min(
                state, exp_name, ch, [p for p, _ in stim_cols_used]
            ) if stim_cols_used else []
            x_labels = [f"{bm:.1f}" for bm in base_min]

            extremum_label = "max" if direction == "increase" else "min"
            window_str = f"stim+{window[0]}…stim+{window[1] - 1} frames"
            cmp_str = "≥" if direction == "increase" else "≤"
            ok = _draw_half_violin_with_box(
                ax=None,
                violin_data=violin_data,
                x_label=x_labels,
                y_label=f"Δ luminosity  ({extremum_label} − baseline)",
                title=(
                    f"{exp_name} / {ch} — responders only "
                    f"(peak Δ dF/F₀ {cmp_str} {sign * dff_threshold:+.4f}, "
                    f"Bonferroni per-cell α=0.01, N={len(stim_frames)};  "
                    f"{n_resp} of {n_total} cells)"
                ),
                save_path=fig_path(exp_name, f"{ch}_response_violin_responders"),
                x_axis_label="Stimulus onset (min)",
            )
            if not ok:
                print(
                    f"{exp_name} / {ch}: no non-empty responder data — skipped."
                )
            else:
                print(
                    f"{exp_name} / {ch}: responder violin — "
                    f"{n_resp}/{n_total} cells passed |dF/F0| ≥ {dff_threshold}."
                )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_per_stimulus_response_violins(experiments, state)
    plot_per_stimulus_response_violins(experiments, state, metric="width")
    plot_per_stimulus_response_violins(
        experiments, state, pool_channels=True, metric="height",
    )
    plot_per_stimulus_response_violins(
        experiments, state, pool_channels=True, metric="width",
    )
    thresholds = compute_responder_thresholds(experiments, state, alpha=0.01)
    plot_per_stimulus_response_violins_responders(
        experiments, state, thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
