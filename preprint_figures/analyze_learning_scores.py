#!/usr/bin/env python3
"""Learning-score analysis (no plotting) → analysis_cache/<exp>/learning_scores.pkl.

Computes the figure-ready intermediates the learning-score figures display, for
DMSO (``response_direction == "increase"``) experiments only:

  * per metric {height, width}: a habituation + sensitization blob holding the
    summed per-cell scores, the permutation null matrices (n_perm × n_cells),
    per-cell one-tailed p-values, BH-FDR q-values, the population-level
    permutation test, and the per-channel population breakdown.
  * a single metric-independent anticipation blob: per train {1, 2}, the real /
    shuffled per-cell rest-region z-scores, the null matrix, and the channel
    index.

This is a near-verbatim port of the source ``learning_scores.py``'s
``compute_learning_scores`` + ``_compute_anticipation_blob`` (and their helpers)
with ALL matplotlib stripped. The *_null matrices are cached because the
permutation-test figures need them. Seeds match the source exactly
(habituation rng_seed=42, sensitization rng_seed=43; anticipation seed_base =
1000*ch_ix + 100*train_idx, overlay = seed_base, null = seed_base + 1).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.baseline_window import (
    per_cell_response_delta_with_baseline,
    prestim_baseline_values,
)
from common.cli import parse_args
from common.config import LEARNING_STIMS_PER_TRAIN, cell_line_label
from common.io_paths import save_analysis_cache
from common.permutation_null import (
    permutation_null_distribution,
    pvalue_one_tailed,
)
from common.pipeline import prepare_state
from common.stats import bh_fdr, population_permutation_pvalue
from common.stim_helpers import compute_stim_caps
from common.time_axis import frames_to_min, response_window_frames

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


# Pre-stim baseline window used for every per-stim Δ — matches the responder
# gate in common/responders.py so the descriptive figures and the responder
# classification share the same baseline definition.
_PRESTIM_BASELINE_FRAMES = 5


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
        baseline = prestim_baseline_values(
            mat, sc, n_pre=_PRESTIM_BASELINE_FRAMES,
        )
        d, w = per_cell_response_delta_with_baseline(
            mat, sc, direction, window, baseline,
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
                             window, n_trains):
    """Resolve train ``t``'s rest region and anticipation column.

    Returns ``(ref_lo, ref_hi, anticip_col)`` or ``None`` when the train is
    not scoreable. ``[ref_lo, ref_hi)`` is the post-train rest region;
    ``anticip_col`` is the single-frame anticipation time (10 min after the
    train's last response peak).

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
    # Need at least 2 frames so the shuffled pick can be different from the
    # real anticipation frame.
    if ref_hi - ref_lo < 2:
        return None

    anticip_min = float(all_minutes[peak_col]) + 10.0
    anticip_col = int(np.argmin(np.abs(all_minutes - anticip_min)))
    if anticip_col < ref_lo or anticip_col >= ref_hi:
        return None
    return ref_lo, ref_hi, anticip_col


def _rest_region_zscores(mat, ref_lo, ref_hi):
    """Per-cell z-score of ``mat`` over the rest region ``[ref_lo, ref_hi)``.

    Each cell is centred and scaled by its own mean and std across the rest
    region; cells with zero (or all-NaN) variance get NaN z-scores.
    """
    rest = mat[:, ref_lo:ref_hi]
    means = np.nanmean(rest, axis=1, keepdims=True)
    stds = np.nanstd(rest, axis=1, keepdims=True)
    stds_safe = np.where(stds == 0, np.nan, stds)
    return (rest - means) / stds_safe


def _compute_anticipation_deltas(z_rest, real_offset, start_pool_offsets, *,
                                 rng):
    """Per-cell real and shuffled rest-region z-scores for one train.

    ``z_rest`` is the per-cell z-scored rest region (shape
    ``(n_cells, rest_len)``). ``real`` is the z-score at the anticipation
    offset; ``shuffled`` is the z-score at a single random other rest-region
    offset.
    """
    n_cells = z_rest.shape[0]
    real = z_rest[:, int(real_offset)].copy()
    if start_pool_offsets.size == 0:
        return real, np.full(n_cells, np.nan, dtype=float)
    picks = start_pool_offsets[
        rng.integers(0, start_pool_offsets.size, size=n_cells)
    ]
    shuffled = z_rest[np.arange(n_cells), picks]
    return real, shuffled


def _anticipation_null_matrix(z_rest, start_pool_offsets, *, n_perm, rng_seed):
    """Null matrix ``(n_perm, n_cells)`` of shuffled per-cell z-scores.

    Each iteration draws one random rest-region offset per cell from
    ``start_pool_offsets`` and records the z-score there. Returns NaN when the
    rest region admits no valid shuffled pick.
    """
    n_cells = z_rest.shape[0]
    rng = np.random.default_rng(rng_seed)
    if start_pool_offsets.size == 0:
        return np.full((n_perm, n_cells), np.nan, dtype=float)
    null = np.empty((n_perm, n_cells), dtype=float)
    cell_ix = np.arange(n_cells)
    for i in range(n_perm):
        picks = start_pool_offsets[
            rng.integers(0, start_pool_offsets.size, size=n_cells)
        ]
        null[i] = z_rest[cell_ix, picks]
    return null


def _compute_anticipation_blob(state, exp_name, cfg, *, n_perm=10000):
    """Per-train rest-region z-score anticipation values for one experiment.

    Anticipation is metric-independent so it is computed once per experiment.
    For each train, each cell's rest-region signal is z-scored (subtract
    per-cell rest-region mean, divide by per-cell rest-region std); the
    "real" z-score is taken 10 min after the train's last response peak, the
    "shuffled" z-score at a random other frame in the same rest region.

    Returns a nested blob keyed by train index (1, 2):

        blob["channel_names"] -> list[str]
        blob["trains"][train_idx] = {
            "channel_index": (n_cells,) int,
            "real":     (n_cells,) float,
            "shuffled": (n_cells,) float,
            "null":     (n_perm, n_cells) float,
        }

    or ``None`` when no channel yields data. ``trains[t]`` is ``None`` when
    every channel failed to score that train.
    """
    direction = cfg.get("response_direction", "increase")
    train_idxs = (1, 2)
    accum = {
        ti: {"real": [], "shuffled": [], "null": [], "channel_index": []}
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
        stim_cols = inputs["stim_cols"]
        window = inputs["window"]
        n_cells, n_cols = mat.shape
        n_trains = len(stim_cols) // LEARNING_STIMS_PER_TRAIN
        all_minutes = inputs["frame_to_min_fn"](np.arange(n_cols))

        for train_idx in train_idxs:
            t = train_idx - 1
            spec = _anticipation_train_spec(
                t, stim_cols, mat, all_minutes, direction, window, n_trains,
            )
            if spec is None:
                print(
                    f"    anticipation: {exp_name} / {ch} — "
                    f"train {train_idx} not scorable "
                    f"(rest region out of range)."
                )
                continue
            ref_lo, ref_hi, anticip_col = spec
            z_rest = _rest_region_zscores(mat, ref_lo, ref_hi)
            real_offset = anticip_col - ref_lo
            rest_offsets = np.arange(ref_hi - ref_lo, dtype=int)
            start_pool_offsets = rest_offsets[rest_offsets != real_offset]

            seed_base = 1000 * ch_ix + 100 * train_idx
            overlay_rng = np.random.default_rng(seed_base)
            real, shuffled = _compute_anticipation_deltas(
                z_rest, real_offset, start_pool_offsets, rng=overlay_rng,
            )
            null = _anticipation_null_matrix(
                z_rest, start_pool_offsets,
                n_perm=n_perm, rng_seed=seed_base + 1,
            )
            accum[train_idx]["real"].append(real)
            accum[train_idx]["shuffled"].append(shuffled)
            accum[train_idx]["null"].append(null)
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
            "real": np.concatenate(entries["real"]),
            "shuffled": np.concatenate(entries["shuffled"]),
            "null": np.concatenate(entries["null"], axis=1),
        }
    blob = {"channel_names": list(used_channels), "trains": trains_blob}
    n_pooled_train1 = (
        trains_blob[1]["real"].size if trains_blob.get(1) is not None else 0
    )
    n_pooled_train2 = (
        trains_blob[2]["real"].size if trains_blob.get(2) is not None else 0
    )
    print(
        f"  anticipation (rest-region z-score): {exp_name} — "
        f"train1 n={n_pooled_train1}, train2 n={n_pooled_train2}, "
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


def analyze(experiments, state, *, n_perm=10000):
    """Compute + cache the learning-score bundle for each DMSO experiment.

    ``compute_learning_scores`` already filters to
    ``response_direction == "increase"`` (DMSO); one ``learning_scores.pkl`` is
    written per such experiment. The cached ``data`` is the per-experiment
    sub-dict ``{"height": blob, "width": blob, "anticipation": blob}`` —
    verbatim, INCLUDING the ``*_null`` matrices the permutation-test figures
    need.
    """
    scores = compute_learning_scores(experiments, state, n_perm=n_perm)
    for exp_name, by_key in scores.items():
        cfg = experiments[exp_name]
        # Channel count for the inferential caveat: prefer a metric blob's
        # used-channel list (height, then width), fall back to anticipation.
        n_channels = 0
        for metric in ("height", "width"):
            blob = by_key.get(metric)
            if blob is not None:
                n_channels = len(blob.get("channel_names", []))
                break
        if n_channels == 0:
            ablob = by_key.get("anticipation")
            if ablob is not None:
                n_channels = len(ablob.get("channel_names", []))
        meta = {
            "exp_name": exp_name,
            "n_channels": int(n_channels),
            "cell_line": cell_line_label(exp_name),
        }
        save_analysis_cache(by_key, exp_name, "learning_scores", meta=meta)
        print(
            f"  cached learning_scores.pkl for {exp_name} "
            f"({n_channels} channels)"
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
