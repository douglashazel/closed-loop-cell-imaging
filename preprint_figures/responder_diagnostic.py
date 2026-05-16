#!/usr/bin/env python3
"""Responder-call diagnostics — is the responder rate biology or artifact?

Per (experiment, channel) this writes two figures into the experiment's
results folder so the PDF aggregator picks them up:

* ``responder_distribution_diagnostic`` — every cell's aggregate per-stim
  Δ dF/F0 as a marginal histogram + jittered strip scatter, with the
  responder threshold drawn through it. A clean responder population is a
  bulk piled at 0 with a separated right tail; an artifact shows up as a
  whole-population blob shifted off 0 that the threshold merely slices.

* ``responder_stimlock_diagnostic`` — the key open check: the population
  Δ dF/F0 aligned to real stim onsets vs. randomly placed pseudo-stims. A
  field-wide step at real stims (absent at pseudo-stims) means a
  stimulus-locked nuisance, not per-cell responses. A footer panel lists
  the diagnoses still outstanding before the responder logic is changed.

Diagnostic only — changes nothing in the responder pipeline.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.io_paths import fig_path
from common.pipeline import prepare_state
from common.responders import (
    _aggregate,
    _channel_dff,
    _delta_at,
    compute_responder_thresholds,
)
from common.stim_helpers import compute_f0_baseline
from common.time_axis import response_window_frames

PLOT_PARAMS = {
    "dpi": 300,
    "title_fontsize": 13,
    "title_fontweight": "bold",
    "suptitle_fontsize": 15,
    "panel_fontsize": 11,
    "nonresponder_color": "#9aa0a6",
    "responder_color": "#e74c3c",
    "threshold_color": "#111111",
    "zero_color": "#888888",
    "real_color": "#363fe9",
    "pseudo_color": "#9aa0a6",
    "window_shade": "#363fe9",
    "hist_alpha": 0.55,
    "scatter_alpha": 0.35,
    "scatter_size": 11,
    "band_alpha": 0.25,
    "bins": 70,
}

ALPHA = 0.01
BASELINE_N_PRE = 5
STAT = "mean"
N_PSEUDO_TRACE = 400
RNG_SEED = 42


def _per_cell_stat(dff_mat, frame_to_col, stim_frames, direction, window):
    """Aggregate per-stim Δ dF/F0 per cell — mirrors compute_responder_masks."""
    per_stim = [
        _delta_at(dff_mat, frame_to_col[p], direction, window, BASELINE_N_PRE)
        for p in stim_frames
        if p in frame_to_col
    ]
    return _aggregate(np.vstack(per_stim), STAT)


def _valid_pseudo_cols(dff_mat, stim_cols, window, exclusion_pad=10):
    """Stimulus-free columns usable as pseudo-stims — mirrors responders.py."""
    n_cols = dff_mat.shape[1]
    win_lo, win_hi = window
    excluded = np.zeros(n_cols, dtype=bool)
    pad = max(exclusion_pad, win_hi)
    for sc in stim_cols:
        excluded[max(0, sc - pad):min(n_cols, sc + pad + 1)] = True
    valid = ~excluded
    valid[: max(0, -win_lo)] = False
    valid[max(0, n_cols - win_hi + 1):] = False
    valid[:BASELINE_N_PRE] = False
    return np.where(valid)[0]


def _aligned_population_trace(dff_mat, anchor_cols, offsets, direction):
    """Population-*median* Δ dF/F0 (vs per-cell pre-stim baseline) by offset.

    The median over cells is used on purpose: a population *mean* is pulled
    up by the minority of genuine strong responders, so it rises even with
    no field-wide nuisance. The median tracks the *typical* cell, which is
    what a stimulus-locked field-wide shift would move.

    Returns ``(mean_over_anchors, sem_over_anchors)`` each of shape
    ``(len(offsets),)`` — averaged across the supplied anchor columns.
    """
    n_cols = dff_mat.shape[1]
    per_anchor = []
    for ac in anchor_cols:
        lo = ac - BASELINE_N_PRE
        if lo < 0 or ac + offsets[-1] >= n_cols or ac + offsets[0] < 0:
            continue
        base = np.nanmean(dff_mat[:, lo:ac], axis=1)
        row = [np.nanmedian(dff_mat[:, ac + o] - base) for o in offsets]
        per_anchor.append(row)
    arr = np.asarray(per_anchor, dtype=float)
    if arr.size == 0:
        return None, None
    n = arr.shape[0]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0) / np.sqrt(max(n, 1))


REMAINING_CHECKS = [
    "1.  Stim-locked population shift  — this figure: does the field-wide "
    "Δ dF/F0 step up at REAL stims but not at pseudo-stims?",
    "2.  Dead-frame proximity  — PC3 masks 24 dead frames (18 flashes); "
    "confirm none sit inside a stim's baseline/response window and bias Δ.",
    "3.  Perfusion / optical artifact  — inspect bg_trace and focus around "
    "stim onsets; a medium-swap refractive shift would lift every cell.",
    "4.  F0 dependence  — check whether per-cell Δ dF/F0 correlates with F0 "
    "(brightness); dimmer PC3 cells should not preferentially pass.",
    "5.  Decision (after 1-4)  — if a field-wide nuisance is confirmed, "
    "score responders relative to the per-stim population median.",
    "6.  alpha sensitivity sweep  — regenerate responder counts at "
    "alpha in {0.001, 0.01, 0.05} once the nuisance is handled.",
]


def _draw_distribution_figure(exp_name, channels, panels):
    """Marginal histogram + jittered strip scatter of per-cell Δ dF/F0."""
    n = len(channels)
    fig = plt.figure(figsize=(6.2 * n, 8.4), dpi=PLOT_PARAMS["dpi"])
    gs = fig.add_gridspec(2, n, height_ratios=[1, 3], hspace=0.07, wspace=0.26,
                          top=0.84, bottom=0.09, left=0.09, right=0.97)
    rng = np.random.default_rng(RNG_SEED)

    for col, ch in enumerate(channels):
        d = panels[ch]
        stat, thr, signed_t, mask = d["stat"], d["thr"], d["signed_t"], d["mask"]
        good = stat[~np.isnan(stat)]
        lo, hi = np.nanpercentile(good, 0.5), np.nanpercentile(good, 99.5)
        clip = np.clip(good, lo, hi)
        resp_clip = np.clip(stat[mask], lo, hi)
        non_clip = np.clip(stat[~mask & ~np.isnan(stat)], lo, hi)

        ax_h = fig.add_subplot(gs[0, col])
        ax_h.hist(non_clip, bins=PLOT_PARAMS["bins"], range=(lo, hi),
                  color=PLOT_PARAMS["nonresponder_color"],
                  alpha=PLOT_PARAMS["hist_alpha"], label="non-responder")
        ax_h.hist(resp_clip, bins=PLOT_PARAMS["bins"], range=(lo, hi),
                  color=PLOT_PARAMS["responder_color"],
                  alpha=PLOT_PARAMS["hist_alpha"], label="responder")
        ax_h.axvline(signed_t, color=PLOT_PARAMS["threshold_color"], lw=2,
                     ls="--", label=f"threshold {signed_t:+.3f}")
        ax_h.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_h.set_title(
            f"{exp_name} — {ch}\n{100 * d['pct_resp']:.1f}% responders  "
            f"(non-resp median Δ = {d['med_non']:+.3f})",
            fontsize=PLOT_PARAMS["title_fontsize"],
            fontweight=PLOT_PARAMS["title_fontweight"])
        ax_h.set_ylabel("cell count")
        ax_h.tick_params(labelbottom=False)
        ax_h.spines[["top", "right"]].set_visible(False)
        ax_h.legend(fontsize=8);

        ax_s = fig.add_subplot(gs[1, col], sharex=ax_h)
        for sel, color, lbl in (
            (~mask & ~np.isnan(stat), PLOT_PARAMS["nonresponder_color"], "non-responder"),
            (mask, PLOT_PARAMS["responder_color"], "responder"),
        ):
            x = np.clip(stat[sel], lo, hi)
            y = rng.uniform(0.0, 1.0, size=x.size)
            ax_s.scatter(x, y, s=PLOT_PARAMS["scatter_size"], color=color,
                         alpha=PLOT_PARAMS["scatter_alpha"], edgecolors="none",
                         label=f"{lbl} (n={int(sel.sum())})")
        ax_s.axvline(signed_t, color=PLOT_PARAMS["threshold_color"], lw=2, ls="--")
        ax_s.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_s.set_xlabel("per-cell aggregate per-stim Δ dF/F0  "
                        "(x clipped to [p0.5, p99.5])")
        ax_s.set_ylabel("jitter (no meaning)")
        ax_s.set_yticks([])
        ax_s.spines[["top", "right", "left"]].set_visible(False)
        ax_s.legend(fontsize=8, loc="upper right");

    fig.suptitle(
        f"{exp_name} — responder distribution diagnostic\n"
        "clean = bulk at 0 with a separated tail   |   "
        "suspect = whole-population blob shifted off 0, threshold slices its shoulder",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=0.97);
    out = fig_path(exp_name, "responder_distribution_diagnostic")
    fig.savefig(out, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_stimlock_figure(exp_name, channels, panels):
    """Stim-aligned population trace + per-stim deltas + remaining-checks panel."""
    n = len(channels)
    height = 4.2 * n + 3.0
    fig = plt.figure(figsize=(13.5, height), dpi=PLOT_PARAMS["dpi"])
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[*([3] * n), 2.4],
                          hspace=0.62, wspace=0.30,
                          top=1.0 - 0.95 / height, bottom=0.05,
                          left=0.08, right=0.97)

    for row, ch in enumerate(channels):
        d = panels[ch]
        offsets = d["offsets"]

        ax_t = fig.add_subplot(gs[row, 0])
        for key, color, lbl in (
            ("real", PLOT_PARAMS["real_color"], "real stims"),
            ("pseudo", PLOT_PARAMS["pseudo_color"], "pseudo-stims"),
        ):
            m, s = d[f"{key}_trace"], d[f"{key}_sem"]
            if m is None:
                continue
            ax_t.plot(offsets, m, color=color, lw=2, label=lbl)
            ax_t.fill_between(offsets, m - s, m + s, color=color,
                              alpha=PLOT_PARAMS["band_alpha"])
        ax_t.axvline(0.0, color=PLOT_PARAMS["zero_color"], lw=1)
        ax_t.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        ax_t.axvspan(d["win_lo"], d["win_hi"], color=PLOT_PARAMS["window_shade"],
                     alpha=0.08, label="response window")
        ax_t.set_title(f"{ch} — population-median Δ dF/F0 aligned to stim onset",
                       fontsize=PLOT_PARAMS["title_fontsize"],
                       fontweight=PLOT_PARAMS["title_fontweight"])
        ax_t.set_xlabel("frame offset from stim onset")
        ax_t.set_ylabel("population-median Δ dF/F0\n(vs per-cell pre-stim baseline)")
        ax_t.spines[["top", "right"]].set_visible(False)
        ax_t.legend(fontsize=8, loc="upper left");

        ax_b = fig.add_subplot(gs[row, 1])
        n_real = d["per_stim_real"].size
        xs = np.arange(1, n_real + 1)
        nb = d["null_band"]
        ax_b.fill_between([0.5, n_real + 0.5], nb[0], nb[2],
                          color=PLOT_PARAMS["pseudo_color"],
                          alpha=PLOT_PARAMS["band_alpha"],
                          label="pseudo-stim null (p1-p99)")
        ax_b.axhline(nb[1], color=PLOT_PARAMS["pseudo_color"], lw=1.2,
                     ls="--", label="pseudo-stim median")
        ax_b.axhline(0.0, color=PLOT_PARAMS["zero_color"], lw=0.8, ls=":")
        ax_b.scatter(xs, d["per_stim_real"], s=44,
                     color=PLOT_PARAMS["real_color"], zorder=3,
                     label="real stim")
        ax_b.set_title(f"{ch} — per-stim population-median Δ dF/F0",
                       fontsize=PLOT_PARAMS["title_fontsize"],
                       fontweight=PLOT_PARAMS["title_fontweight"])
        ax_b.set_xlabel("stimulus index")
        ax_b.set_ylabel("population-median Δ dF/F0")
        ax_b.set_xticks(xs)
        ax_b.spines[["top", "right"]].set_visible(False)
        ax_b.legend(fontsize=7, loc="lower right");

    ax_c = fig.add_subplot(gs[n, :])
    ax_c.axis("off")
    ax_c.text(0.0, 1.0, "Remaining diagnoses before changing the responder logic",
              fontsize=PLOT_PARAMS["title_fontsize"],
              fontweight=PLOT_PARAMS["title_fontweight"], va="top")
    ax_c.text(0.0, 0.84, "\n".join(REMAINING_CHECKS),
              fontsize=PLOT_PARAMS["panel_fontsize"], va="top", family="monospace")

    fig.suptitle(
        f"{exp_name} — stimulus-locked artifact check\n"
        "if the real-stim trace steps up where the pseudo-stim trace stays flat, "
        "the responder rate is inflated by a field-wide nuisance",
        fontsize=PLOT_PARAMS["suptitle_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"], y=1.0 - 0.30 / height);
    out = fig_path(exp_name, "responder_stimlock_diagnostic")
    fig.savefig(out, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    thresholds = compute_responder_thresholds(
        experiments, state, alpha=ALPHA, baseline_n_pre=BASELINE_N_PRE, stat=STAT,
    )
    rng = np.random.default_rng(RNG_SEED)

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        sign = -1.0 if direction == "decrease" else 1.0
        channels = list(cfg["channels"])
        panels = {}

        for ch in channels:
            window = response_window_frames(state, exp_name, ch, cfg)
            win_lo, win_hi = window
            dff_mat, frame_to_col = _channel_dff(state, exp_name, ch, cfg)
            stim_frames = cfg["stim_frames"][ch]
            stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]

            stat = _per_cell_stat(dff_mat, frame_to_col, stim_frames,
                                  direction, window)
            thr = float(thresholds.get((exp_name, ch), 0.10))
            signed_t = sign * thr
            mask = ((stat <= signed_t) if direction == "decrease"
                    else (stat >= signed_t)) & ~np.isnan(stat)
            good = stat[~np.isnan(stat)]
            non = good[(good < signed_t) if direction != "decrease"
                       else (good > signed_t)]

            valid_cols = _valid_pseudo_cols(dff_mat, stim_cols, window)
            offsets = np.arange(-(BASELINE_N_PRE + 2), win_hi + 4)
            real_trace, real_sem = _aligned_population_trace(
                dff_mat, stim_cols, offsets, direction)
            if valid_cols.size:
                pcs = rng.choice(valid_cols, size=N_PSEUDO_TRACE,
                                 replace=valid_cols.size < N_PSEUDO_TRACE)
                pseudo_trace, pseudo_sem = _aligned_population_trace(
                    dff_mat, list(pcs), offsets, direction)
            else:
                pseudo_trace = pseudo_sem = None

            per_stim_real = np.array([
                np.nanmedian(_delta_at(dff_mat, sc, direction, window,
                                       BASELINE_N_PRE))
                for sc in stim_cols
            ])
            if valid_cols.size:
                null = np.array([
                    np.nanmedian(_delta_at(dff_mat, int(pc), direction, window,
                                           BASELINE_N_PRE))
                    for pc in rng.choice(valid_cols, size=N_PSEUDO_TRACE,
                                         replace=valid_cols.size < N_PSEUDO_TRACE)
                ])
                null_band = (np.nanpercentile(null, 1),
                             np.nanpercentile(null, 50),
                             np.nanpercentile(null, 99))
            else:
                null_band = (0.0, 0.0, 0.0)

            panels[ch] = {
                "stat": stat, "thr": thr, "signed_t": signed_t, "mask": mask,
                "pct_resp": float(mask.sum()) / max(good.size, 1),
                "med_non": float(np.median(non)) if non.size else float("nan"),
                "offsets": offsets, "win_lo": win_lo, "win_hi": win_hi,
                "real_trace": real_trace, "real_sem": real_sem,
                "pseudo_trace": pseudo_trace, "pseudo_sem": pseudo_sem,
                "per_stim_real": per_stim_real, "null_band": null_band,
            }
            f0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
            print(f"  {exp_name} / {ch}: {100 * panels[ch]['pct_resp']:.1f}% "
                  f"responders | non-resp median Δ={panels[ch]['med_non']:+.4f} "
                  f"| median F0={np.nanmedian(f0):.1f}")

        d_path = _draw_distribution_figure(exp_name, channels, panels)
        s_path = _draw_stimlock_figure(exp_name, channels, panels)
        print(f"  → {d_path}")
        print(f"  → {s_path}")


if __name__ == "__main__":
    main()
