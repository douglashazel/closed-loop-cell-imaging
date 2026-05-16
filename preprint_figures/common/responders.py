"""Per-(experiment, channel) responder threshold + mask computation.

Responders are scored by the *mean* (default) or *median* per-stim
Δ dF/F0 across **all** of an experiment's stimuli — not the single best
pulse. Because that aggregate is one test per cell, no Bonferroni
correction over the stimulus count is applied: the null is built from
the same aggregation over randomly placed pseudo-stimuli, so averaging
N pulses tightens the null by ~sqrt(N) instead of pushing the threshold
deeper into the tail. A cell that responds consistently but modestly to
every pulse now passes; a single noisy spike no longer does.

Each per-stim Δ dF/F0 uses a multi-frame pre-stim baseline (mean of the
``baseline_n_pre`` frames immediately before the stim) instead of the
single stim-frame value. The single-frame baseline was a major source
of the heavy null tail on the noisier datasets (PC3, C2C12); averaging a
short pre-stim window removes it.
"""

import sys

import numpy as np

from common.baseline_window import (
    per_cell_response_delta_with_baseline,
    prestim_baseline_values,
)
from common.stim_helpers import compute_f0_baseline
from common.time_axis import response_window_frames

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def _aggregate(stacked, stat):
    """Collapse a ``(n_stim, n_cells)`` delta stack to ``(n_cells,)``."""
    if stat == "median":
        return np.nanmedian(stacked, axis=0)
    return np.nanmean(stacked, axis=0)


def _channel_dff(state, exp_name, ch, cfg):
    """Return ``(dff_mat, frame_to_col)`` for one (experiment, channel).

    ``dff_mat`` rows follow the corrected-luminosity ``CellID`` order, so
    callers can index trace matrices and responder masks interchangeably.
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

    F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff_mat = (mat - F0) / F0_safe
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    return dff_mat, frame_to_col


def _delta_at(dff_mat, stim_col, direction, window, baseline_n_pre):
    """Per-cell Δ dF/F0 at one stim column, vs. a pre-stim-window baseline."""
    baseline = prestim_baseline_values(dff_mat, stim_col, n_pre=baseline_n_pre)
    return per_cell_response_delta_with_baseline(
        dff_mat, stim_col, direction, window, baseline,
    )


def compute_responder_thresholds(
    experiments, state,
    alpha=0.01,
    exclusion_pad=10,
    n_pseudo=1000,
    baseline_n_pre=5,
    stat="mean",
    rng_seed=42,
):
    """Per-(experiment, channel) responder threshold on the aggregate statistic.

    The responder statistic for a cell is the ``stat`` ("mean" or
    "median") of its per-stim Δ dF/F0 across all ``N_real`` stimuli. The
    null distribution of that *same* statistic is built by drawing
    ``N_real`` pseudo-stimuli from stimulus-free columns, aggregating
    their per-cell deltas, and repeating ``n_pseudo`` times. The
    threshold is the ``100·(1 − alpha)``-th percentile of the null
    ``|statistic|`` — one aggregate test per cell, hence no Bonferroni
    correction over the stimulus count.

    Returns ``{(exp_name, ch_name): threshold_magnitude}``.
    """
    rng = np.random.default_rng(rng_seed)
    thresholds = {}

    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")

        for ch in cfg["channels"]:
            window = response_window_frames(state, exp_name, ch, cfg)
            win_lo, win_hi = window
            stim_frames = cfg["stim_frames"][ch]
            n_real = len(stim_frames)
            if n_real == 0:
                continue

            dff_mat, frame_to_col = _channel_dff(state, exp_name, ch, cfg)
            n_cols = dff_mat.shape[1]

            stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
            excluded = np.zeros(n_cols, dtype=bool)
            pad = max(exclusion_pad, win_hi)
            for sc in stim_cols:
                lo = max(0, sc - pad)
                hi = min(n_cols, sc + pad + 1)
                excluded[lo:hi] = True
            valid_mask = ~excluded
            valid_mask[: max(0, -win_lo)] = False
            valid_mask[max(0, n_cols - win_hi + 1):] = False
            # A pseudo-stim needs `baseline_n_pre` frames of headroom for
            # its own pre-stim baseline window.
            valid_mask[:baseline_n_pre] = False
            valid_cols = np.where(valid_mask)[0]

            pct = 100.0 * (1.0 - alpha)

            def _fallback(reason):
                thresholds[(exp_name, ch)] = 0.10
                print(
                    f"  responder threshold ({exp_name} / {ch}): "
                    f"|{stat} Δ dF/F0| ≥ 0.1000  (fallback: {reason}; "
                    f"N_real={n_real}, α={alpha:g})"
                )

            if len(valid_cols) < 1:
                _fallback("no stimulus-free columns")
                continue

            replace = len(valid_cols) < n_real
            ch_null = []
            for _ in range(n_pseudo):
                pcs = rng.choice(valid_cols, size=n_real, replace=replace)
                per_stim = np.vstack([
                    _delta_at(dff_mat, int(pc), direction, window, baseline_n_pre)
                    for pc in pcs
                ])
                ch_null.append(_aggregate(per_stim, stat))

            ch_abs = np.abs(np.concatenate(ch_null))
            ch_abs = ch_abs[~np.isnan(ch_abs)]
            if ch_abs.size == 0:
                _fallback("empty null")
                continue

            threshold = float(np.nanpercentile(ch_abs, pct))
            thresholds[(exp_name, ch)] = threshold
            print(
                f"  responder threshold ({exp_name} / {ch}): "
                f"|{stat} Δ dF/F0| ≥ {threshold:.4f}  "
                f"({pct:.2f}th pct of {stat}-of-{n_real} pseudo-stim null, "
                f"N_real={n_real}, α={alpha:g}, baseline={baseline_n_pre} frames)"
            )

    return thresholds


def compute_responder_masks(
    experiments, state, thresholds=None, *,
    alpha=0.01, baseline_n_pre=5, stat="mean",
):
    """Per-(experiment, channel) boolean responder mask over cells.

    A cell is a responder when its aggregate per-stim Δ dF/F0 — the
    ``stat`` ("mean" or "median") across every stimulus — crosses the
    channel's threshold (signed by ``response_direction``: ``≥`` for an
    ``"increase"``, ``≤`` for a ``"decrease"``). Returns
    ``{(exp_name, ch): bool ndarray}`` with one entry per cell, aligned
    to the corrected-luminosity row order so callers can index trace
    matrices directly.

    ``thresholds`` is the dict from :func:`compute_responder_thresholds`;
    when omitted it is computed here with the same ``alpha``,
    ``baseline_n_pre`` and ``stat``.
    """
    if thresholds is None:
        thresholds = compute_responder_thresholds(
            experiments, state, alpha=alpha,
            baseline_n_pre=baseline_n_pre, stat=stat,
        )

    masks = {}
    for exp_name, cfg in experiments.items():
        direction = cfg.get("response_direction", "increase")
        sign = -1.0 if direction == "decrease" else 1.0

        for ch in cfg["channels"]:
            window = response_window_frames(state, exp_name, ch, cfg)
            stim_frames = cfg["stim_frames"][ch]
            dff_mat, frame_to_col = _channel_dff(state, exp_name, ch, cfg)
            n_cells = dff_mat.shape[0]

            per_stim = []
            for p in stim_frames:
                if p not in frame_to_col:
                    continue
                per_stim.append(
                    _delta_at(
                        dff_mat, frame_to_col[p], direction, window,
                        baseline_n_pre,
                    )
                )
            if not per_stim:
                masks[(exp_name, ch)] = np.zeros(n_cells, dtype=bool)
                continue

            per_cell_stat = _aggregate(np.vstack(per_stim), stat)
            signed_t = sign * float(thresholds.get((exp_name, ch), 0.10))
            if direction == "decrease":
                mask = per_cell_stat <= signed_t
            else:
                mask = per_cell_stat >= signed_t
            masks[(exp_name, ch)] = mask & ~np.isnan(per_cell_stat)

    return masks
