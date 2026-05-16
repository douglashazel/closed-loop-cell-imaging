#!/usr/bin/env python3
"""Per-stimulus response violins (asymmetric half-violin + notched box).

Per experiment:
    * pooled_response_violin_height_dff.png  — pooled per-stim Δ dF/F0 height
    * pooled_response_violin_width_dff.png   — pooled per-stim response width

Each figure carries a train-level repeated-measures test (cell-level Friedman
within the experiment + a replicate-level one-sample t-test across channels)
and overlays each channel's per-train mean as a biological-replicate point.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import LEARNING_STIMS_PER_TRAIN, PEAK_OFFSET
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stats import friedman_with_posthoc, inferential_caveat, one_sample_t_dz
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


def _sig_stars(p):
    """Significance label for a p-value: ``***`` / ``**`` / ``*`` / ``ns``.

    Returns ``"n/a"`` when ``p`` is missing or non-finite (e.g. only one
    biological replicate, so no replicate-level test was run).
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _draw_replicate_train_inset(ax_, chan_means, train_p):
    """Draw the per-replicate train-mean inset in the upper-right corner.

    ``chan_means`` has shape ``(n_channels, n_trains)`` — one green line per
    biological replicate across the stimulus trains, ramped dark→light.
    ``train_p`` is the replicate-level one-sample t-test p-value for the
    first→last train change; its significance is annotated as a bracket
    spanning train 1 to train N (``*`` / ``**`` / ``***`` or ``ns``).
    """
    n_ch, n_tr = chan_means.shape
    axin = ax_.inset_axes([0.66, 0.66, 0.32, 0.30])
    train_x = np.arange(n_tr)
    greens = PLOT_PARAMS["replicate_greens"]
    for ci in range(n_ch):
        axin.plot(
            train_x, chan_means[ci],
            color=greens[ci % len(greens)],
            linewidth=1.8, marker="D", markersize=6,
            markeredgecolor="#222222", markeredgewidth=0.5,
            label=f"Rep {ci + 1}",
        )
    axin.set_xticks(train_x)
    axin.set_xticklabels(
        [f"Train {t + 1}" for t in range(n_tr)], fontsize=8,
    )
    axin.tick_params(labelsize=8)
    axin.spines[["top", "right"]].set_visible(False)
    axin.set_title(
        "Per-replicate train mean", fontsize=9,
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    axin.set_xlabel("Stimulus train", fontsize=8)
    axin.set_ylabel("Mean response", fontsize=8)

    # Significance bracket for the first→last train change (replicate level).
    if n_tr >= 2:
        y0, y1 = axin.get_ylim()
        span = (y1 - y0) or 1.0
        bar_y = y1 + span * 0.10
        axin.plot(
            [train_x[0], train_x[-1]], [bar_y, bar_y],
            color="#222222", linewidth=1.2,
        )
        axin.text(
            (train_x[0] + train_x[-1]) / 2.0, bar_y + span * 0.05,
            _sig_stars(train_p),
            ha="center", va="bottom", fontsize=11,
            fontweight=PLOT_PARAMS["title_fontweight"],
        )
        axin.set_ylim(y0, bar_y + span * 0.30)
    axin.legend(fontsize=7, loc="best", framealpha=0.85);


def _draw_half_violin_with_box(ax, violin_data, x_label, y_label, title, save_path,
                                x_axis_label="Peak frame",
                                stats_text=None, chan_means=None, caveat=None,
                                train_p=None):
    """Render the asymmetric violin/box composite for one figure and save it.

    ``stats_text`` (the train-level repeated-measures result) is drawn as an
    upper-left box; ``chan_means`` of shape ``(n_channels, n_trains)`` is drawn
    as an upper-right inset of each biological replicate's per-train mean, with
    ``train_p`` driving the inset's significance annotation; ``caveat`` is the
    figure footnote naming the unit of inference.
    """
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

    ax_.legend(fontsize=PLOT_PARAMS["legend_fontsize_large"], loc="center right");

    # Per-replicate (per-channel) train means shown in a small upper-right
    # inset, so the biological-replicate structure behind the train-level test
    # is visible without crowding the pooled cell-level violins.
    if chan_means is not None and chan_means.size:
        _draw_replicate_train_inset(ax_, chan_means, train_p)

    if stats_text:
        ax_.text(
            0.01, 0.99, stats_text,
            ha="left", va="top", transform=ax_.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"] - 1,
            family="monospace", zorder=10,
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.92),
        )

    ax_.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax_.set_xticks(range(n_cat))
    ax_.set_xticklabels([str(lbl) for lbl in x_label], fontsize=9)
    ax_.set_xlabel(x_axis_label, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax_.set_ylabel(y_label, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    plt.tight_layout(rect=(0, 0.035, 1, 1));
    if caveat:
        fig.text(
            0.5, 0.008, caveat, ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
    fig.savefig(save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)
    return True


def _build_signal_matrix(state, exp_name, ch, cfg, *, signal):
    """Return ``(values_by_col, df_indexed, frame_cols, frame_to_col)`` for the chosen signal.

    ``signal == "lum"`` returns the raw corrected luminosity matrix; ``"dff"``
    returns ``(mat - F0) / F0`` so amplitude and width violins can be plotted
    in normalized units.
    """
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
    if signal == "dff":
        F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
        F0_safe = np.where(F0 == 0, np.nan, F0)
        values = (mat - F0) / F0_safe
    else:
        values = mat
    return values, df_indexed, frame_cols, frame_to_col


def _per_channel_stim_response_arrays(
    state, exp_name, ch, cfg, *, metric, signal="lum",
):
    """Return ``(violin_data, base_min, n_total)`` for one (exp, channel).

    ``signal`` selects between raw luminosity (default) and dF/F0; both produce
    the same return shapes so the caller can swap freely.
    """
    direction = cfg.get("response_direction", "increase")
    window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))

    stim_frames = cfg["stim_frames"][ch]
    values, df_indexed, frame_cols, frame_to_col = _build_signal_matrix(
        state, exp_name, ch, cfg, signal=signal,
    )
    n_cells, n_cols = values.shape

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
                values, col, direction, window,
                return_width=True,
                cap_col=cap_for_col[col],
                frame_to_min_fn=_f2m,
            )
            vals = widths[~np.isnan(widths)]
        else:
            deltas = per_cell_response_delta(values, col, direction, window)
            vals = deltas[~np.isnan(deltas)]
        violin_data.append(vals)

    base_min = list(frames_to_min(state, exp_name, ch, stim_frames)) if stim_frames else []
    return violin_data, base_min, n_cells


def _per_train_cell_means(state, exp_name, ch, cfg, *, metric, signal,
                          n_per_train=LEARNING_STIMS_PER_TRAIN):
    """Per-cell mean response within each stimulus train for one channel.

    Returns ``(cell_train_means, n_trains)`` where ``cell_train_means`` has
    shape ``(n_cells, n_trains)`` — entry (c, t) is cell c's NaN-mean response
    over the ``n_per_train`` pulses of train t. Returns ``(None, 0)`` when the
    channel's stimuli do not divide into at least two whole trains (e.g. the
    acid experiment, whose pulses are not organized into fixed trains).
    """
    direction = cfg.get("response_direction", "increase")
    window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
    stim_frames = cfg["stim_frames"][ch]
    n_stims = len(stim_frames)
    n_trains = n_stims // n_per_train
    if n_stims == 0 or n_stims % n_per_train != 0 or n_trains < 2:
        return None, 0

    values, _, _, frame_to_col = _build_signal_matrix(
        state, exp_name, ch, cfg, signal=signal,
    )
    n_cells, n_cols = values.shape

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

    # Per-cell response per stim, keeping cell alignment and NaNs (a missing
    # stim leaves an all-NaN row; nanmean over the train then ignores it).
    per_stim = np.full((n_stims, n_cells), np.nan)
    for i, p in enumerate(stim_frames):
        if p not in frame_to_col:
            continue
        col = frame_to_col[p]
        if metric == "width":
            _, widths = per_cell_response_delta(
                values, col, direction, window,
                return_width=True, cap_col=cap_for_col[col],
                frame_to_min_fn=_f2m,
            )
            per_stim[i] = widths
        else:
            per_stim[i] = per_cell_response_delta(values, col, direction, window)

    trains = per_stim.reshape(n_trains, n_per_train, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN train
        cell_train_means = np.nanmean(trains, axis=1).T  # (n_cells, n_trains)
    return cell_train_means, n_trains


def _train_level_stats(per_channel_means, *, metric, n_trains):
    """Cell-level Friedman + replicate-level one-sample t across stim trains.

    ``per_channel_means`` is a list of ``(n_cells, n_trains)`` arrays (or None),
    one per channel — channels are the biological replicates. Two layers are
    reported: a within-experiment Friedman repeated-measures test treating
    cells as observations (with Holm-corrected Wilcoxon post-hoc per train
    pair), and a replicate-level one-sample t-test of the per-channel
    last-minus-first train difference against 0. Returns
    ``(stats_text, chan_train_means, train_p)`` — ``chan_train_means`` has
    shape ``(n_channels, n_trains)`` for the inset and ``train_p`` is the
    replicate-level p-value for that difference — or ``(None, None, None)``.
    """
    arrays = [a for a in per_channel_means if a is not None]
    if not arrays or n_trains < 2:
        return None, None, None

    pooled = np.concatenate(arrays, axis=0)  # (total_cells, n_trains)
    n_total = pooled.shape[0]
    complete = pooled[~np.isnan(pooled).any(axis=1)]
    n_complete = complete.shape[0]

    lines = [f"Train-level comparison ({metric};  T1 … T{n_trains})"]

    # Cell-level layer — within-experiment, cells are NOT biological replicates.
    fr = friedman_with_posthoc(complete)
    if fr.get("insufficient"):
        lines.append(
            f"cell-level: only {n_complete} cells tracked across all "
            f"trains — too few for Friedman"
        )
    else:
        lines.append(
            f"cell-level (within-expt; {n_complete}/{n_total} cells tracked "
            f"across all trains):"
        )
        lines.append(
            f"  Friedman chi2={fr['friedman_stat']:.1f}, "
            f"p={fr['friedman_p']:.3g}, Kendall W={fr['kendall_w']:.3f}"
        )
        for ph in fr["posthoc"]:
            i, j = ph["pair"]
            lines.append(
                f"  T{i + 1}-T{j + 1}: Wilcoxon p={ph['p_adj']:.3g} (Holm), "
                f"rank-biserial r={ph['rank_biserial']:+.3f}"
            )

    # Replicate-level layer — channels are the biological replicates.
    chan_means = np.vstack([np.nanmean(a, axis=0) for a in arrays])
    n_ch = chan_means.shape[0]
    diffs = chan_means[:, -1] - chan_means[:, 0]
    train_p = None
    if n_ch >= 2:
        t = one_sample_t_dz(diffs)
        train_p = t["p_value"]
        per_ch = "  ".join(f"{d:+.3f}" for d in diffs)
        lines.append(
            f"replicate-level ({n_ch} channels) "
            f"deltaT{n_trains}-T1 per replicate = [{per_ch}]"
        )
        lines.append(
            f"  1-sample t vs 0: p={t['p_value']:.3g}, "
            f"Cohen dz={t['cohen_dz']:+.2f}"
        )
    else:
        lines.append(
            "replicate-level: 1 channel — within-experiment only, "
            "no biological-replicate test"
        )
    # The overlay shows across-replicate spread; with a single channel there
    # are no replicate points to plot, so suppress it (the lone line would
    # just be the pooled per-train cell mean, not a replicate mean).
    return "\n".join(lines), (chan_means if n_ch >= 2 else None), train_p


def plot_per_stimulus_response_violins(
    experiments, state, *, metric="height", signal="dff",
):
    """Pooled-across-channels violin of per-cell response per stim.

    ``signal`` selects ``"lum"`` (raw corrected luminosity) or ``"dff"``
    (dF/F0, default). When ``signal == "dff"`` the saved filenames append a
    ``_dff`` suffix.
    """
    suffix = "_dff" if signal == "dff" else ""
    signal_label = "dF/F₀" if signal == "dff" else "Δ luminosity"
    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        window = cfg.get("response_window", (PEAK_OFFSET, PEAK_OFFSET + 1))
        extremum_label = "max" if direction == "increase" else "min"
        window_str = f"stim+{window[0]}…stim+{window[1] - 1} frames"
        if metric == "width":
            unit_str = "(dF/F₀ baseline)" if signal == "dff" else ""
            y_label = f"Response width (min) {unit_str}".strip()
            width_cap_note = (
                " — width search capped at min inter-stim interval"
            )
        else:
            y_label = f"{signal_label}  ({extremum_label} − baseline)"
            width_cap_note = ""

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
        per_channel_train_means = []
        train_n = 0
        for ch in channels:
            vd, base_min, n_cells = _per_channel_stim_response_arrays(
                state, exp_name, ch, cfg, metric=metric, signal=signal,
            )
            ctm, n_tr = _per_train_cell_means(
                state, exp_name, ch, cfg, metric=metric, signal=signal,
            )
            per_channel_train_means.append(ctm)
            train_n = max(train_n, n_tr)
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
        stats_text, chan_means, train_p = _train_level_stats(
            per_channel_train_means, metric=metric, n_trains=train_n,
        )
        caveat = inferential_caveat(exp_name, len(channels), unit="cell")
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
                exp_name, f"pooled_response_violin_{metric}{suffix}"
            ),
            x_axis_label="Stimulus onset (min)",
            stats_text=stats_text,
            chan_means=chan_means,
            caveat=caveat,
            train_p=train_p,
        )
        if not ok:
            print(
                f"{exp_name}: pooled response violin ({metric}) — "
                "no non-empty data; skipped."
            )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    plot_per_stimulus_response_violins(
        experiments, state, metric="height", signal="dff",
    )
    plot_per_stimulus_response_violins(
        experiments, state, metric="width", signal="dff",
    )


if __name__ == "__main__":
    main()
