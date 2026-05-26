#!/usr/bin/env python3
"""Learning-score histograms (DMSO experiments only).

Per DMSO experiment, the habituation / sensitization figures (× height/width)
are unchanged: each is overlaid with a permutation null and a stats box
reporting the FDR-corrected per-cell hit count plus a population-level
permutation test (pooled and per biological replicate):
    * learning_habituation_height.png
    * learning_habituation_width.png
    * learning_sensitization_height.png
    * learning_sensitization_width.png

Anticipation is now scored per anticipation period (post-train 1 and
post-train 2) independently on the dF/F₀-normalized signal. The rest region
for train ``t`` runs from the end of train ``t``'s last response to the start
of train ``t+1``'s first stim. For each cell we record the dF/F₀ value at the
anticipation time (10 min after the train's last response peak) and the
dF/F₀ value at a random other frame in the rest region. Two plots per train,
per variant (no-window vs. 5-frame window picking the largest dF/F₀):
    * learning_anticipation_train{1,2}{,_window}.png
    * learning_anticipation_train{1,2}{,_window}_permtest.png
The first overlays the real (blue) and shuffled (gray) distributions; the
second is a permutation-mean test with 10 000 shuffles and a two-tailed p.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import LEARNING_STIMS_PER_TRAIN
from common.io_paths import fig_path, save_fig
from common.permutation_null import permutation_null_distribution, pvalue_one_tailed
from common.pipeline import prepare_state
from common.plot_params import PLOT_PARAMS
from common.stats import bh_fdr, inferential_caveat, population_permutation_pvalue
from common.stim_helpers import (
    compute_f0_baseline,
    compute_stim_caps,
    per_cell_response_delta,
)
from common.time_axis import frames_to_min, response_window_frames

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


def _build_learning_inputs(state, exp_name, ch, cfg, *, metric):
    """Per-channel inputs needed by every learning-score function."""
    direction = cfg.get("response_direction", "increase")
    window = response_window_frames(state, exp_name, ch, cfg)
    stim_frames = cfg["stim_frames"][ch]
    df_indexed = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df_indexed.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    if not frame_cols:
        return None
    frame_nums = [int(str(c).lstrip("f")) for c in frame_cols]
    mat = df_indexed[frame_cols].values
    n_cells, n_cols = mat.shape
    frame_to_col = {f: i for i, f in enumerate(frame_nums)}
    valid_stim_cols = [frame_to_col[p] for p in stim_frames if p in frame_to_col]
    if not valid_stim_cols:
        return None
    caps = compute_stim_caps(valid_stim_cols, n_cols)

    def f2m(frames):
        return frames_to_min(state, exp_name, ch, frames)

    per_stim_height = np.full((len(valid_stim_cols), n_cells), np.nan)
    per_stim_width = np.full((len(valid_stim_cols), n_cells), np.nan)
    for i, (sc, cap) in enumerate(zip(valid_stim_cols, caps)):
        d, w = per_cell_response_delta(
            mat, sc, direction, window,
            return_width=True, cap_col=cap, frame_to_min_fn=f2m,
        )
        per_stim_height[i] = d
        per_stim_width[i] = w

    per_stim = per_stim_width if metric == "width" else per_stim_height
    return {
        "mat": mat,
        "stim_cols": valid_stim_cols,
        "frame_to_min_fn": f2m,
        "per_stim": per_stim,
        "n_cells": n_cells,
        "direction": direction,
        "window": window,
        "cell_ids": list(df_indexed.index),
        "exp_name": exp_name,
        "ch": ch,
    }


def _score_running_extremum(per_stim, *, mode, n_per_train=LEARNING_STIMS_PER_TRAIN):
    """Habituation (``mode='min'``) or sensitization (``mode='max'``).

    Returns ``(summed, per_train)`` where ``summed.shape == (n_cells,)`` and
    ``per_train.shape == (n_trains, n_cells)``. ``summed == per_train.sum(0)``.
    """
    n_stims, n_cells = per_stim.shape
    n_trains = n_stims // n_per_train
    per_train = np.zeros((n_trains, n_cells), dtype=int)
    for t in range(n_trains):
        train = per_stim[t * n_per_train:(t + 1) * n_per_train]
        if mode == "min":
            current = np.full(n_cells, np.inf)
        else:
            current = np.full(n_cells, -np.inf)
        seen_any = np.zeros(n_cells, dtype=bool)
        for s in range(n_per_train):
            r = train[s]
            valid = ~np.isnan(r)
            if mode == "min":
                beats = valid & (r < current)
            else:
                beats = valid & (r > current)
            update_score = beats & seen_any
            update_value = beats | (valid & ~seen_any)
            per_train[t] += update_score.astype(int)
            current = np.where(update_value, r, current)
            seen_any = seen_any | valid
    return per_train.sum(axis=0), per_train


def _anticipation_train_spec(t, stim_cols, mat, all_minutes, direction,
                             window, n_trains, n_win=5):
    """Resolve train ``t``'s rest region, anticipation column, and 5-frame window.

    Returns ``(ref_lo, ref_hi, anticip_col, win_cols)`` or ``None`` when the
    train is not scoreable. ``[ref_lo, ref_hi)`` is the post-train rest region;
    ``anticip_col`` is the single-frame anticipation time (10 min after the
    train's last response peak); ``win_cols`` is a length-``n_win`` integer
    range centred on ``anticip_col``.

    The rest region runs from the end of train ``t``'s last response to the
    start of train ``t+1``'s first stim. The last train of the experiment
    therefore has no scoreable rest region under this definition and is
    skipped (returns ``None``).
    """
    if t >= n_trains - 1:
        return None
    n_cols = mat.shape[1]
    win_lo_off, win_hi_off = window
    train = stim_cols[
        t * LEARNING_STIMS_PER_TRAIN:(t + 1) * LEARNING_STIMS_PER_TRAIN
    ]
    last_sc = train[-1]
    next_first_sc = stim_cols[(t + 1) * LEARNING_STIMS_PER_TRAIN]

    # Last response peak of the train: median peak column across cells.
    pk_lo = max(0, last_sc + win_lo_off)
    pk_hi = min(n_cols, last_sc + win_hi_off)
    if pk_lo >= pk_hi:
        return None
    seg = mat[:, pk_lo:pk_hi]
    if direction == "decrease":
        peak_offsets = np.argmin(np.where(np.isnan(seg), np.inf, seg), axis=1)
    else:
        peak_offsets = np.argmax(np.where(np.isnan(seg), -np.inf, seg), axis=1)
    peak_col = int(np.round(np.nanmedian(pk_lo + peak_offsets)))

    # Rest region: end of last response → start of next train's first stim.
    end_resp_col = min(last_sc + win_hi_off, n_cols - 1)
    ref_lo = end_resp_col
    ref_hi = int(next_first_sc)
    if ref_hi - ref_lo < n_win:
        return None

    # Anticipation window: 10 min after the peak, ± n_win // 2 frames.
    anticip_min = float(all_minutes[peak_col]) + 10.0
    anticip_col = int(np.argmin(np.abs(all_minutes - anticip_min)))
    half = n_win // 2
    win_cols = np.arange(anticip_col - half, anticip_col - half + n_win)

    # Both the single anticipation frame and the full window must sit inside
    # the rest region (the null draws windows from within that region).
    if anticip_col < ref_lo or anticip_col >= ref_hi:
        return None
    if win_cols[0] < ref_lo or win_cols[-1] >= ref_hi:
        return None
    return ref_lo, ref_hi, anticip_col, win_cols


def _sliding_window_max(dff, n_win):
    """Per-cell max in every length-``n_win`` sliding window along axis 1.

    Returns ``(n_cells, n_cols - n_win + 1)``. ``n_win == 1`` is the identity.
    """
    if n_win == 1:
        return np.asarray(dff)
    return np.nanmax(
        np.lib.stride_tricks.sliding_window_view(dff, n_win, axis=1), axis=2,
    )


def _anticipation_start_pool(ref_lo, ref_hi, real_start, n_win):
    """Valid shuffled window-start columns within ``[ref_lo, ref_hi)``.

    Every start ``s`` is required to keep the full window ``[s, s + n_win)``
    inside the rest region. The real start (``anticip_col`` for the
    no-window variant, ``win_cols[0]`` for the 5-frame variant) is excluded.
    """
    hi = ref_hi - n_win + 1
    if hi <= ref_lo:
        return np.empty(0, dtype=int)
    candidates = np.arange(ref_lo, hi, dtype=int)
    return candidates[candidates != int(real_start)]


def _compute_anticipation_deltas(win_max, start_pool, real_start, *, rng):
    """Per-cell real and shuffled dF/F₀ deltas for one train / variant.

    ``win_max`` is the precomputed sliding-window max (or the dFF matrix when
    ``n_win == 1``). ``real`` is the per-cell value at the real start;
    ``shuffled`` is the per-cell value at a single random start drawn from
    ``start_pool``.
    """
    n_cells = win_max.shape[0]
    real = win_max[:, int(real_start)].copy()
    if start_pool.size == 0:
        return real, np.full(n_cells, np.nan, dtype=float)
    picks = start_pool[rng.integers(0, start_pool.size, size=n_cells)]
    shuffled = win_max[np.arange(n_cells), picks]
    return real, shuffled


def _anticipation_null_matrix(win_max, start_pool, *, n_perm, rng_seed):
    """Null matrix ``(n_perm, n_cells)`` of shuffled per-cell dF/F₀ deltas.

    Each iteration draws one random start per cell from ``start_pool`` and
    records the corresponding ``win_max`` value. Returns NaN when the rest
    region admits no valid shuffled start.
    """
    n_cells = win_max.shape[0]
    rng = np.random.default_rng(rng_seed)
    if start_pool.size == 0:
        return np.full((n_perm, n_cells), np.nan, dtype=float)
    null = np.empty((n_perm, n_cells), dtype=float)
    cell_ix = np.arange(n_cells)
    for i in range(n_perm):
        picks = start_pool[rng.integers(0, start_pool.size, size=n_cells)]
        null[i] = win_max[cell_ix, picks]
    return null


def _compute_anticipation_blob(state, exp_name, cfg, *, n_perm=10000):
    """Per-train, per-variant dF/F₀ anticipation deltas for one experiment.

    Anticipation is metric-independent so it is computed once per experiment.
    Returns a nested blob keyed by train index (1, 2) and variant
    (``"no_window"`` / ``"window"``):

        blob["channel_names"] -> list[str]
        blob["trains"][train_idx] = {
            "channel_index": (n_cells,) int,
            "no_window": {"real": ..., "shuffled": ..., "null": ...},
            "window":    {"real": ..., "shuffled": ..., "null": ...},
        }

    or ``None`` when no channel yields data. ``trains[t]`` is ``None`` when
    every channel failed to score that train.
    """
    direction = cfg.get("response_direction", "increase")
    train_idxs = (1, 2)
    accum = {
        ti: {
            "no_window": {"real": [], "shuffled": [], "null": []},
            "window":    {"real": [], "shuffled": [], "null": []},
            "channel_index": [],
        }
        for ti in train_idxs
    }
    used_channels = []

    for ch_ix, ch in enumerate(cfg["channels"]):
        inputs = _build_learning_inputs(
            state, exp_name, ch, cfg, metric="height",
        )
        if inputs is None:
            continue
        used_channels.append(ch)
        mat = inputs["mat"]
        F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
        # F0 has shape (n_cells, 1); reuse the lum_dict_to_df row order that
        # _build_learning_inputs already used, so rows align with mat.
        F0_safe = np.where(F0 == 0, np.nan, F0)
        dff = (mat - F0_safe) / F0_safe
        win_max_singles = _sliding_window_max(dff, 1)
        win_max_5 = _sliding_window_max(dff, 5)

        stim_cols = inputs["stim_cols"]
        window = inputs["window"]
        n_cells, n_cols = mat.shape
        n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
        all_minutes = inputs["frame_to_min_fn"](np.arange(n_cols))

        for t_idx, train_idx in enumerate(train_idxs):
            t = train_idx - 1
            spec = _anticipation_train_spec(
                t, stim_cols, mat, all_minutes, direction, window, n_trains,
            )
            if spec is None:
                print(
                    f"    anticipation: {exp_name} / {ch} — "
                    f"train {train_idx} not scorable "
                    f"(rest region / window out of range)."
                )
                continue
            ref_lo, ref_hi, anticip_col, win_cols = spec
            # Each variant has its own RNG seed so the per-cell shuffles for
            # the overlay plot are independent of the null draws below.
            seed_base = 1000 * ch_ix + 100 * train_idx
            for variant_key, n_win, real_start, win_max in (
                ("no_window", 1, anticip_col, win_max_singles),
                ("window",    5, int(win_cols[0]), win_max_5),
            ):
                start_pool = _anticipation_start_pool(
                    ref_lo, ref_hi, real_start, n_win,
                )
                overlay_rng = np.random.default_rng(
                    seed_base + (10 if variant_key == "window" else 0),
                )
                real, shuffled = _compute_anticipation_deltas(
                    win_max, start_pool, real_start, rng=overlay_rng,
                )
                null = _anticipation_null_matrix(
                    win_max, start_pool, n_perm=n_perm,
                    rng_seed=seed_base + (11 if variant_key == "window" else 1),
                )
                accum[train_idx][variant_key]["real"].append(real)
                accum[train_idx][variant_key]["shuffled"].append(shuffled)
                accum[train_idx][variant_key]["null"].append(null)
            accum[train_idx]["channel_index"].append(
                np.full(n_cells, ch_ix, dtype=int)
            )

    if not used_channels:
        return None
    trains_blob = {}
    for train_idx in train_idxs:
        entries = accum[train_idx]
        if not entries["channel_index"]:
            trains_blob[train_idx] = None
            continue
        trains_blob[train_idx] = {
            "channel_index": np.concatenate(entries["channel_index"]),
            "no_window": {
                "real": np.concatenate(entries["no_window"]["real"]),
                "shuffled": np.concatenate(entries["no_window"]["shuffled"]),
                "null": np.concatenate(entries["no_window"]["null"], axis=1),
            },
            "window": {
                "real": np.concatenate(entries["window"]["real"]),
                "shuffled": np.concatenate(entries["window"]["shuffled"]),
                "null": np.concatenate(entries["window"]["null"], axis=1),
            },
        }
    blob = {"channel_names": list(used_channels), "trains": trains_blob}
    n_pooled_train1 = (
        trains_blob[1]["no_window"]["real"].size
        if trains_blob.get(1) is not None else 0
    )
    n_pooled_train2 = (
        trains_blob[2]["no_window"]["real"].size
        if trains_blob.get(2) is not None else 0
    )
    print(
        f"  anticipation (dF/F₀): {exp_name} — train1 n={n_pooled_train1}, "
        f"train2 n={n_pooled_train2}, "
        f"channels={len(used_channels)}."
    )
    return blob


def _add_population_stats(blob, channel_index, n_channels):
    """Augment a learning blob with FDR q-values + population-level tests.

    Per-cell permutation p-values answer "which cells learn" but, pooled over
    hundreds of cells, need an FDR correction (``<measure>_qvalue``) before the
    significant count means anything. The population-level question — "does the
    cell *population* learn more than under shuffled stim order" — is answered
    by comparing the observed mean score to the per-permutation mean of the
    existing null (``<measure>_pop``), and again within each channel /
    biological replicate (``<measure>_pop_by_channel``) so a single dish
    driving the pooled result stays visible.
    """
    for m in ("habituation", "sensitization"):
        pvals = blob.get(f"{m}_pvalue")
        null = blob.get(f"{m}_null")
        if pvals is None or null is None:
            continue
        blob[f"{m}_qvalue"] = bh_fdr(pvals)
        blob[f"{m}_pop"] = population_permutation_pvalue(blob[m], null)
        per_channel = []
        for ci in range(n_channels):
            sel = channel_index == ci
            per_channel.append(
                population_permutation_pvalue(blob[m][sel], null[:, sel])
                if np.any(sel) else None
            )
        blob[f"{m}_pop_by_channel"] = per_channel


def compute_learning_scores(experiments, state, *, n_perm=10000):
    """Compute habituation + sensitization + anticipation scores per DMSO expt.

    Returns a nested dict ``out[exp]`` keyed by ``"height"`` / ``"width"``
    (a habituation + sensitization blob per metric) and ``"anticipation"`` (a
    single metric-independent anticipation blob). Each blob holds the observed
    scores plus, when ``n_perm > 0``, the permutation null distributions and
    one-tailed per-cell p-values. Experiments are filtered to
    ``response_direction == "increase"`` (DMSO).
    """
    out = {}
    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue
        out[exp_name] = {}
        for metric in ("height", "width"):
            hab_chunks, sen_chunks = [], []
            hab_null_chunks, sen_null_chunks = [], []
            used_channels = []
            for ch in cfg["channels"]:
                inputs = _build_learning_inputs(
                    state, exp_name, ch, cfg, metric=metric,
                )
                if inputs is None:
                    continue
                used_channels.append(ch)
                hab, _ = _score_running_extremum(inputs["per_stim"], mode="min")
                sen, _ = _score_running_extremum(inputs["per_stim"], mode="max")
                hab_chunks.append(hab)
                sen_chunks.append(sen)

                if n_perm and n_perm > 0:
                    hab_null = permutation_null_distribution(
                        lambda ps: _score_running_extremum(ps, mode="min")[0],
                        inputs["per_stim"], n_perm=n_perm,
                        rng_seed=42, mode="per_cell",
                    )
                    sen_null = permutation_null_distribution(
                        lambda ps: _score_running_extremum(ps, mode="max")[0],
                        inputs["per_stim"], n_perm=n_perm,
                        rng_seed=43, mode="per_cell",
                    )
                    hab_null_chunks.append(hab_null)
                    sen_null_chunks.append(sen_null)

            if not hab_chunks:
                out[exp_name][metric] = None
                continue
            blob = {
                "habituation": np.concatenate(hab_chunks),
                "sensitization": np.concatenate(sen_chunks),
            }
            # Keep replicate identity: which channel each pooled cell came
            # from (channels are biological replicates). Null matrices below
            # are concatenated in the same channel order, so this one index
            # aligns the score vectors and the null columns alike.
            blob["channel_index"] = np.concatenate(
                [np.full(len(c), ci, dtype=int)
                 for ci, c in enumerate(hab_chunks)]
            )
            blob["channel_names"] = list(used_channels)
            if hab_null_chunks:
                blob["habituation_null"] = np.concatenate(hab_null_chunks, axis=1)
                blob["sensitization_null"] = np.concatenate(sen_null_chunks, axis=1)
                blob["habituation_pvalue"] = pvalue_one_tailed(
                    blob["habituation"], blob["habituation_null"],
                )
                blob["sensitization_pvalue"] = pvalue_one_tailed(
                    blob["sensitization"], blob["sensitization_null"],
                )
                _add_population_stats(
                    blob, blob["channel_index"], len(used_channels),
                )
            out[exp_name][metric] = blob
            print(
                f"  learning scores: {exp_name} ({metric}) — "
                f"{len(blob['habituation'])} cells pooled across "
                f"{len(used_channels)} channels."
            )
        out[exp_name]["anticipation"] = _compute_anticipation_blob(
            state, exp_name, cfg, n_perm=n_perm,
        )
    return out


def _format_score_stats(qvalues, pop_result, pop_by_channel, q_alpha=0.05):
    """Compose the multi-line stats annotation for a learning histogram.

    Reports the FDR-significant cell count (per-cell q-values), the
    population-level permutation test (p + z effect size) and, when more than
    one biological replicate is present, the per-replicate z so a single dish
    driving the pooled result stays visible.
    """
    lines = []
    if qvalues is not None and np.size(qvalues):
        finite = np.isfinite(qvalues)
        n_sig = int(np.sum(qvalues[finite] < q_alpha))
        lines.append(f"FDR q<{q_alpha}: {n_sig}/{int(finite.sum())} cells")
    if pop_result is not None:
        lines.append(
            f"population: p={pop_result['p_value']:.3g}, "
            f"z={pop_result['z']:.2f}"
        )
    if pop_by_channel:
        parts = [
            f"c{ci + 1} z={res['z']:.1f}"
            for ci, res in enumerate(pop_by_channel) if res is not None
        ]
        if len(parts) > 1:
            lines.append("per replicate: " + "  ".join(parts))
    return "\n".join(lines)


def _plot_score_histogram(scores, *, title, xlabel, save_path,
                          bins=None, color=None, null_dist=None, x_max=None,
                          qvalues=None, pop_result=None, pop_by_channel=None,
                          caveat=None):
    """Single-distribution histogram with optional permutation-null overlay.

    When ``null_dist`` (shape ``(n_perm, n_cells)``) is provided, plot the
    pooled null as a back-layer histogram normalized to the same total cell
    count. The stats box reports FDR-corrected per-cell hits plus the
    population-level test (see :func:`_format_score_stats`); ``caveat`` is
    drawn as a figure footnote naming the unit of inference.

    When ``x_max`` is given the score axis is fixed to the discrete whole
    numbers ``0..x_max``.
    """
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    if bins is None:
        if x_max is not None:
            max_v = int(x_max)
        else:
            max_v = int(np.nanmax(scores)) if scores.size else 0
            if null_dist is not None and null_dist.size:
                max_v = max(max_v, int(np.nanmax(null_dist)))
        bins = np.arange(-0.5, max_v + 1.5, 1)
    if null_dist is not None and null_dist.size:
        n_perm = null_dist.shape[0]
        ax.hist(
            null_dist.ravel(), bins=bins,
            color="#7a7a7a", alpha=0.55,
            edgecolor="#444444", linewidth=0.4,
            weights=np.full(null_dist.size, 1.0 / max(n_perm, 1)),
            label=f"Shuffled null (mean over {n_perm} perms)",
            zorder=1,
        )
    ax.hist(
        scores, bins=bins,
        color=color or PLOT_PARAMS["fit_color"],
        alpha=0.55, edgecolor="#222222", linewidth=0.6,
        label=f"Observed (n={scores.size})",
        zorder=2,
    )
    stats_text = _format_score_stats(qvalues, pop_result, pop_by_channel)
    if stats_text:
        ax.text(
            0.98, 0.95, stats_text,
            ha="right", va="top",
            transform=ax.transAxes,
            fontsize=PLOT_PARAMS["legend_fontsize"],
            bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
        )
    if x_max is not None:
        ax.set_xticks(np.arange(0, int(x_max) + 1))
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    if null_dist is not None and null_dist.size:
        ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best")
    plt.tight_layout(rect=(0, 0.035, 1, 1));
    if caveat:
        fig.text(
            0.5, 0.008, caveat, ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
    save_fig(fig, save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def _plot_anticipation_dff_histogram(real, shuffled, *, title, save_path,
                                      caveat=None):
    """Real (blue) vs. shuffled (gray) dF/F₀ anticipation deltas.

    Both distributions hold one value per cell; shared bins are computed from
    the pooled finite values. Dashed vertical lines mark the two means.
    """
    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)

    pooled = np.concatenate([np.asarray(real), np.asarray(shuffled)])
    finite = pooled[np.isfinite(pooled)]
    if finite.size:
        bins = np.histogram_bin_edges(finite, bins=40)
    else:
        bins = 40
    ax.hist(
        shuffled, bins=bins, color="#7a7a7a", alpha=0.55,
        edgecolor="#444444", linewidth=0.4,
        label=f"Shuffled (n={np.asarray(shuffled).size})",
        zorder=1,
    )
    ax.hist(
        real, bins=bins, color=PLOT_PARAMS["fit_color"], alpha=0.6,
        edgecolor="#222222", linewidth=0.6,
        label=f"Observed (n={np.asarray(real).size})",
        zorder=2,
    )
    mean_real = float(np.nanmean(real))
    mean_shuf = float(np.nanmean(shuffled))
    ax.axvline(
        mean_real, color=PLOT_PARAMS["fit_color"],
        linewidth=1.8, linestyle="--", zorder=3,
    )
    ax.axvline(
        mean_shuf, color="#444444",
        linewidth=1.4, linestyle="--", zorder=3,
    )
    stats_text = (
        f"n cells = {np.asarray(real).size}\n"
        f"mean observed = {mean_real:.3g}\n"
        f"mean shuffled = {mean_shuf:.3g}"
    )
    ax.text(
        0.98, 0.95, stats_text,
        ha="right", va="top", transform=ax.transAxes,
        fontsize=PLOT_PARAMS["legend_fontsize"],
        bbox=dict(facecolor="white", edgecolor="#999999", alpha=0.9),
    )
    ax.set_xlabel(
        "dF/F₀ delta", fontsize=PLOT_PARAMS["axis_label_fontsize"],
    )
    ax.set_ylabel("Cells", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best");
    plt.tight_layout(rect=(0, 0.035, 1, 1));
    if caveat:
        fig.text(
            0.5, 0.008, caveat, ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
    save_fig(fig, save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def _permutation_mean_pvalue(observed, null_mat):
    """Two-tailed permutation p-value for the population mean score.

    ``M_real`` is the observed mean score across cells; ``M_shuffled`` is the
    per-permutation mean of the shuffled-null scores. The p-value is the
    fraction of shuffled means whose absolute deviation from the null mean
    exceeds the observed mean's deviation. Returns
    ``(M_real, M_shuffled, p_value)``.
    """
    M_real = float(np.nanmean(observed))
    M_shuffled = np.nanmean(null_mat, axis=1)
    null_mean = float(np.nanmean(M_shuffled))
    d_real = abs(M_real - null_mean)
    d_shuf = np.abs(M_shuffled - null_mean)
    p_value = float(np.mean(d_shuf > d_real))
    return M_real, M_shuffled, p_value


def _plot_permutation_mean_test(observed, null_mat, *, title, xlabel,
                                save_path, caveat=None):
    """Histogram of shuffled mean scores with the observed mean overlaid.

    Visualizes the permutation test: each cell's response order is shuffled
    ``n_perm`` times, the mean learning score recomputed each time, and the
    distribution of shuffled means (``M_shuffled``) plotted as a histogram with
    a vertical line at the observed mean (``M_real``). The annotated p-value is
    the two-tailed permutation p (see :func:`_permutation_mean_pvalue`).
    """
    M_real, M_shuffled, p_value = _permutation_mean_pvalue(observed, null_mat)
    n_perm = int(M_shuffled.size)

    fig, ax = plt.subplots(
        figsize=PLOT_PARAMS["figsize"], dpi=PLOT_PARAMS["dpi"],
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.hist(
        M_shuffled, bins=40,
        color="#7a7a7a", alpha=0.65,
        edgecolor="#444444", linewidth=0.4,
        label=f"Shuffled mean ({n_perm} perms)",
        zorder=1,
    )
    p_disp = (f"< {1.0 / n_perm:.0e}" if p_value == 0.0
              else f"= {p_value:.4g}")
    ax.axvline(
        M_real, color=PLOT_PARAMS["fit_color"], linewidth=2.4,
        label=(f"Observed mean = {M_real:.3f}\n"
               f"two-tailed permutation p {p_disp}"),
        zorder=3,
    )
    ax.set_xlabel(xlabel, fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_ylabel("Permutations", fontsize=PLOT_PARAMS["axis_label_fontsize"])
    ax.set_title(
        title,
        fontsize=PLOT_PARAMS["title_fontsize"],
        fontweight=PLOT_PARAMS["title_fontweight"],
    )
    ax.legend(fontsize=PLOT_PARAMS["legend_fontsize"], loc="best");
    plt.tight_layout(rect=(0, 0.035, 1, 1));
    if caveat:
        fig.text(
            0.5, 0.008, caveat, ha="center", va="bottom",
            fontsize=PLOT_PARAMS["legend_fontsize"] - 2,
            style="italic", color="#555555",
        )
    save_fig(fig, save_path, dpi=PLOT_PARAMS["dpi"], bbox_inches="tight")
    plt.close(fig)


def plot_learning_score_histograms(experiments, state, scores=None):
    """Emit habituation / sensitization / anticipation histograms (DMSO).

    For habituation / sensitization: ``learning_<measure>_<metric>.png`` per
    metric (unchanged). For anticipation: per train (1, 2) and per variant
    (no-window vs. 5-frame window) — a real-vs-shuffled overlay plus a
    permutation-mean test, on the dF/F₀-normalized signal.
    """
    if scores is None:
        scores = compute_learning_scores(experiments, state)
    for exp_name, by_key in scores.items():
        for metric in ("height", "width"):
            blob = by_key.get(metric)
            if blob is None:
                continue
            n_ch = len(blob.get("channel_names", []))
            caveat = inferential_caveat(exp_name, n_ch, unit="cell")
            for measure_key, label_word in (
                ("habituation", "Habituation"),
                ("sensitization", "Sensitization"),
            ):
                summed = blob[measure_key]
                null = blob.get(f"{measure_key}_null")
                _plot_score_histogram(
                    summed,
                    title=(
                        f"{exp_name} — {label_word.lower()} score ({metric})\n"
                        f"new {'minimums' if measure_key == 'habituation' else 'maximums'} "
                        f"per cell across trains × increments"
                    ),
                    xlabel=f"{label_word} score ({metric})",
                    save_path=fig_path(
                        exp_name, f"learning_{measure_key}_{metric}",
                    ),
                    null_dist=null,
                    qvalues=blob.get(f"{measure_key}_qvalue"),
                    pop_result=blob.get(f"{measure_key}_pop"),
                    pop_by_channel=blob.get(f"{measure_key}_pop_by_channel"),
                    caveat=caveat,
                    x_max=12,
                )
                if null is not None and null.size:
                    _plot_permutation_mean_test(
                        summed, null,
                        title=(
                            f"{exp_name} — {label_word.lower()} permutation "
                            f"test ({metric})\n"
                            f"observed mean score vs shuffled-stim-order null"
                        ),
                        xlabel=f"Mean {label_word.lower()} score ({metric})",
                        save_path=fig_path(
                            exp_name,
                            f"learning_{measure_key}_{metric}_permtest",
                        ),
                        caveat=caveat,
                    )

        ablob = by_key.get("anticipation")
        if ablob is None or not ablob.get("trains"):
            continue
        n_ch = len(ablob.get("channel_names", []))
        caveat = inferential_caveat(exp_name, n_ch, unit="cell")
        for train_idx in (1, 2):
            train_blob = ablob["trains"].get(train_idx)
            if train_blob is None:
                continue
            for variant_key, suffix, win_word in (
                ("no_window", "",        "no window"),
                ("window",    "_window", "5-frame window"),
            ):
                sub = train_blob[variant_key]
                _plot_anticipation_dff_histogram(
                    sub["real"], sub["shuffled"],
                    title=(
                        f"{exp_name} — anticipation "
                        f"(train {train_idx}, {win_word})\n"
                        f"dF/F₀ at anticipation time vs random rest-region times"
                    ),
                    save_path=fig_path(
                        exp_name,
                        f"learning_anticipation_train{train_idx}{suffix}",
                    ),
                    caveat=caveat,
                )
                null = sub.get("null")
                if null is not None and null.size:
                    _plot_permutation_mean_test(
                        sub["real"], null,
                        title=(
                            f"{exp_name} — anticipation permutation test "
                            f"(train {train_idx}, {win_word})\n"
                            f"observed mean dF/F₀ vs shuffled rest-region null"
                        ),
                        xlabel="Mean dF/F₀ delta",
                        save_path=fig_path(
                            exp_name,
                            f"learning_anticipation_train{train_idx}"
                            f"{suffix}_permtest",
                        ),
                        caveat=caveat,
                    )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    scores = compute_learning_scores(experiments, state)
    plot_learning_score_histograms(experiments, state, scores=scores)


if __name__ == "__main__":
    main()
