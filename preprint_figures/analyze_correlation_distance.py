#!/usr/bin/env python3
"""Correlation-vs-distance analysis (no plotting) →
analysis_cache/<exp>/correlation_distance.pkl.

Pairwise correlation vs spatial distance over the FULL dF/F0-normalized time
series. Per experiment, per channel this caches the figure-ready triangle
VECTORS (not the NxN matrices):

  * ``pw_dist``            — upper-triangle pairwise distance (μm via
                             ``PIXELS_PER_UM``)
  * ``pw_corr_by_method``  — {"pearson", "spearman"} upper-triangle correlation
                             vectors over the full dF/F0 time series
  * ``pair_classes``       — {"RR","NN","RN"} 1-D bool masks aligned to the same
                             triu order (None when no responder threshold)
  * ``mantel``             — {method: {"all": result, "RR": result|None}} Mantel
                             permutation tests (n_perm=999, rng_seed=42), the
                             pseudoreplication-safe significance test

Plus the pooled-across-channels combined-figure reductions (replicate-level
Mantel p-values per method per subset, and the combined stat text), and the
precomputed ``inferential_caveat`` strings. The descriptive least-squares fit
line is NOT computed here — it is a deterministic render-side concern recomputed
from the cached vectors in ``plots/correlation_distance.py``.

dF/F0 math is the verbatim ``(mat - F0) / F0_safe`` with F0 from
``compute_f0_baseline`` — identical to the original ``correlation_distance.py``.
No matplotlib / style imports. DMSO + NRK (all experiments).
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_responders import get_responder_masks
from common.cli import parse_args
from common.config import cell_line_label
from common.io_paths import save_analysis_cache
from common.pipeline import prepare_state
from common.stats import inferential_caveat, mantel_test, one_sample_t_dz
from common.stim_helpers import compute_f0_baseline

sys.path.insert(0, "SCRIPTS")
from io_utils import lum_dict_to_df  # noqa: E402


PIXELS_PER_UM = 180.1  # imaging calibration: 0.00555 μm/pixel
MIN_FRAMES_FOR_CORR = 5
METHODS = ("pearson", "spearman")
WINDOW_LABEL = "full dF/F0 time series"


def mean_cell_positions(traj, n_frames):
    """Return ``{cell_id_str: (mean_x, mean_y)}`` averaged over valid frames."""
    positions = {}
    for cid, coords in traj.items():
        xs, ys = [], []
        for i in range(n_frames):
            x = coords.get(f"x{i}")
            y = coords.get(f"y{i}")
            if x is None or y is None:
                continue
            try:
                xs.append(float(x))
                ys.append(float(y))
            except (TypeError, ValueError):
                continue
        if xs:
            positions[cid] = (float(np.mean(xs)), float(np.mean(ys)))
    return positions


def _classify_pair_classes(responder_mask):
    """Return a dict of pair-class → 1-D boolean mask aligned to triu_indices(k=1).

    For ``n`` cells, the upper-triangle iu has ``n*(n-1)/2`` entries. Each
    pair (i, j) is classified as RR (both True), NN (both False), or RN (mix).
    """
    n = len(responder_mask)
    if n < 2:
        return {"RR": np.array([], dtype=bool), "NN": np.array([], dtype=bool), "RN": np.array([], dtype=bool)}
    iu = np.triu_indices(n, k=1)
    a = responder_mask[iu[0]]
    b = responder_mask[iu[1]]
    return {
        "RR": a & b,
        "NN": (~a) & (~b),
        "RN": a ^ b,
    }


def _combined_mantel_stat_text(per_channel_pairs, method):
    """One-line Mantel summary for the pooled fit's legend.

    Each channel contributes its own cell-level Mantel r; those per-replicate
    r values are combined with a one-sample t-test against 0 — the replicate-
    level biological test. A single pooled Mantel test would mix independent
    dishes and has no valid label permutation, so it is not used.
    """
    r_per_channel = []
    for entry in per_channel_pairs:
        res = entry.get("mantel", {}).get(method, {}).get("all")
        if res and not res.get("insufficient"):
            r_per_channel.append(res["r_obs"])
    if not r_per_channel:
        return "Mantel: no channel had enough cells to test"
    if len(r_per_channel) < 2:
        return (
            f"Mantel r={r_per_channel[0]:+.3f} "
            f"(1 replicate — no replicate-level test)"
        )
    combined = one_sample_t_dz(r_per_channel)
    if np.isfinite(combined["p_value"]):
        return (
            f"Mantel (per-channel r vs 0, {combined['n']} replicates): "
            f"mean r={combined['mean']:+.3f}, p={combined['p_value']:.3g}, "
            f"dz={combined['cohen_dz']:+.2f}"
        )
    return f"Mantel: {combined['n']} replicate(s) — too few to test"


def _combined_mantel_pvalue(per_channel_pairs, method, key):
    """Replicate-level Mantel result for one subset (``key`` = "all" or "RR").

    Combines each channel's cell-level Mantel r for ``key`` via a one-sample
    t-test of the per-channel r against 0 — the biological-replicate test that
    underlies the pooled fit line (the ordinary regression p over the pooled
    pairs is invalid; the pairs are not independent). Returns
    ``{"p_value", "mean_r", "n"}`` (``p_value`` is NaN when only one channel was
    testable) or ``None`` when no channel had enough cells.
    """
    r_per_channel = []
    for entry in per_channel_pairs:
        res = entry.get("mantel", {}).get(method, {}).get(key)
        if res and not res.get("insufficient"):
            r_per_channel.append(res["r_obs"])
    if not r_per_channel:
        return None
    if len(r_per_channel) < 2:
        return {"p_value": np.nan, "mean_r": float(r_per_channel[0]), "n": 1}
    combined = one_sample_t_dz(r_per_channel)
    return {
        "p_value": combined["p_value"],
        "mean_r": combined["mean"],
        "n": combined["n"],
    }


def _channel_context(state, exp_name, ch, cfg, responder_masks):
    """Build the per-channel triangle vectors + NxN matrices needed for Mantel.

    Returns ``None`` when there are fewer than 2 positioned cells (the panel is
    "insufficient data" in the source). dF/F0 math + position alignment are
    verbatim from the original ``correlation_distance.py``.
    """
    df = lum_dict_to_df(state["corrected_lum"][exp_name][ch]).set_index("CellID")
    frame_cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    mat = df[frame_cols].values
    # Normalize each cell to its own pre-stim baseline so the correlation is
    # computed on dF/F0 (consistent with the rest of the pipeline). F0 rows
    # align with df.index because compute_f0_baseline reads the same
    # corrected_lum table.
    F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    mat = (mat - F0) / F0_safe
    cell_ids_int = list(df.index)

    positions = mean_cell_positions(
        state["traj_by_channel"][exp_name][ch],
        state["frame_counts"][exp_name][ch],
    )
    keep_rows, pos_xy = [], []
    for r, cid_int in enumerate(cell_ids_int):
        for key in (str(cid_int), cid_int):
            if key in positions:
                keep_rows.append(r)
                pos_xy.append(positions[key])
                break
    pos_xy = np.array(pos_xy, dtype=float)

    if len(pos_xy) < 2:
        return None

    mat_k = mat[keep_rows]
    dist_mat = squareform(pdist(pos_xy, metric="euclidean"))
    dist_sq = dist_mat / PIXELS_PER_UM
    iu = np.triu_indices(len(pos_xy), k=1)
    pw_dist = dist_sq[iu]

    pair_classes = None
    row_mask = None
    # ``responder_masks`` rows follow the corrected-lum CellID order, i.e. the
    # same order as ``cell_ids_int``.
    full_mask = responder_masks.get((exp_name, ch))
    if full_mask is not None:
        row_mask = np.array(
            [bool(full_mask[r]) for r in keep_rows],
            dtype=bool,
        )
        pair_classes = _classify_pair_classes(row_mask)

    return {
        "mat_k": mat_k,
        "pw_dist": pw_dist,
        "dist_sq": dist_sq,
        "iu": iu,
        "n_cells": len(keep_rows),
        "pair_classes": pair_classes,
        "row_mask": row_mask,
    }


def _channel_correlations_and_mantel(ctx):
    """Pearson + Spearman triangle vectors and the Mantel result dicts.

    Returns ``(pw_corr_by_method, mantel_by_method)`` or ``None`` when the
    channel has too few frames (the source draws an "insufficient samples"
    placeholder). The Mantel seeds match the source (n_perm=999, rng_seed=42 via
    the mantel_test default).
    """
    mat_k = ctx["mat_k"]
    if mat_k.shape[1] < MIN_FRAMES_FOR_CORR:
        return None

    pearson_mat = pd.DataFrame(mat_k).T.corr(method="pearson").values
    spearman_mat = pd.DataFrame(mat_k).T.corr(method="spearman").values
    iu = ctx["iu"]
    pw_corr_by_method = {
        "pearson": pearson_mat[iu],
        "spearman": spearman_mat[iu],
    }

    # Mantel test per correlation method. Permuting cell labels makes the
    # *cell* (not the pair) the unit of exchangeability — the pseudoreplication-
    # safe significance test here. The responder sub-matrix gets its own Mantel
    # when >=4 responder cells exist.
    dist_sq = ctx["dist_sq"]
    row_mask = ctx["row_mask"]
    mantel_by_method = {}
    for mth, cmat in (("pearson", pearson_mat), ("spearman", spearman_mat)):
        m_all = mantel_test(dist_sq, cmat, n_perm=999)
        m_rr = None
        if row_mask is not None and int(row_mask.sum()) >= 4:
            rr_idx = np.where(row_mask)[0]
            m_rr = mantel_test(
                dist_sq[np.ix_(rr_idx, rr_idx)],
                cmat[np.ix_(rr_idx, rr_idx)], n_perm=999,
            )
        mantel_by_method[mth] = {"all": m_all, "RR": m_rr}

    return pw_corr_by_method, mantel_by_method


def analyze(experiments, state):
    """Cache the triangle vectors + pair classes + Mantel dicts per experiment.

    Per channel writes the figure-ready vectors; per experiment writes the
    pooled-across-channels combined reductions (replicate-level Mantel p-values
    and the combined stat text) and the precomputed inferential-caveat strings.
    """
    responder_masks = get_responder_masks(experiments, state)

    for exp_name, cfg in experiments.items():
        channels = cfg["channels"]
        cell_line = cell_line_label(exp_name)

        # Per-channel entries, in channel order. ``status`` records the source's
        # placeholder branches ("insufficient data" / "insufficient samples")
        # so the plotting layer can reproduce them without raw data.
        per_channel = []
        per_channel_pairs = []  # only the channels that produced real data
        for ch in channels:
            ctx = _channel_context(state, exp_name, ch, cfg, responder_masks)
            if ctx is None:
                per_channel.append({"ch": ch, "status": "insufficient_data"})
                continue
            corr_mantel = _channel_correlations_and_mantel(ctx)
            if corr_mantel is None:
                per_channel.append({"ch": ch, "status": "insufficient_samples"})
                continue
            pw_corr_by_method, mantel_by_method = corr_mantel

            # NRK encodes the chamber as a trailing letter ("channel 1 A");
            # C2C12/PC3 channels end in a digit. Only the chamber-labelled (NRK)
            # Pearson panels use the caption-oriented title + clean legend.
            chamber = ch.split()[-1]
            is_chamber = bool(chamber.isalpha())

            entry = {
                "ch": ch,
                "status": "ok",
                "n_cells": ctx["n_cells"],
                "pw_dist": np.asarray(ctx["pw_dist"], dtype=float),
                "pw_corr_by_method": {
                    m: np.asarray(v, dtype=float)
                    for m, v in pw_corr_by_method.items()
                },
                "pair_classes": ctx["pair_classes"],
                "mantel": mantel_by_method,
                "chamber": chamber,
                "is_chamber": is_chamber,
            }
            per_channel.append(entry)
            per_channel_pairs.append(entry)

        # Pooled-across-channels combined reductions (replicate-level p-values
        # + the combined stat text), per method. Computed HERE so the plotting
        # layer never recomputes a Mantel test.
        combined = {}
        for method in METHODS:
            combined[method] = {
                "p_rr": _combined_mantel_pvalue(per_channel_pairs, method, "RR"),
                "p_all": _combined_mantel_pvalue(per_channel_pairs, method, "all"),
                "stat_text": _combined_mantel_stat_text(per_channel_pairs, method),
            }

        # Precompute the inferential-caveat footnote strings verbatim. The
        # per-channel and combined figures use slightly different ``extra`` text;
        # n_channels = len(channels) for both (matches the source).
        n_channels = len(channels)
        caveat_per_channel = inferential_caveat(
            exp_name, n_channels, unit="cell pair",
            extra="Significance: Mantel permutation test (per channel); "
                  "slope/r are descriptive.",
        )
        # The combined figure passes n_channels = len(per_channel_pairs) in the
        # source (the count of channels that produced data).
        n_combined = len(per_channel_pairs)
        caveat_combined = inferential_caveat(
            exp_name, n_combined, unit="cell pair",
            extra="Pooled line is descriptive; significance = per-channel "
                  "Mantel test combined across replicates by a one-sample "
                  "t-test of the per-channel Mantel r against 0.",
        )

        data = {
            "per_channel": per_channel,
            "combined": combined,
            "window_label": WINDOW_LABEL,
        }
        meta = {
            "cell_line": cell_line,
            "exp_name": exp_name,
            "window_label": WINDOW_LABEL,
            "n_channels": int(n_channels),
            "n_channels_with_data": int(n_combined),
            "caveat_per_channel": caveat_per_channel,
            "caveat_combined": caveat_combined,
        }
        save_analysis_cache(data, exp_name, "correlation_distance", meta=meta)
        print(
            f"  cached correlation_distance.pkl for {exp_name} "
            f"({n_combined}/{n_channels} channels with data)"
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
