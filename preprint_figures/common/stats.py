"""Significance-test helpers shared across the preprint figures.

The figure pipeline pools many cells across several imaging channels. Channels
are separate dishes/wells, so they are the *biological replicates*; cells
within a channel are nested/technical observations and pulses within a cell
are repeated measures. These helpers let each figure report tests at the
right level:

* ``friedman_with_posthoc`` / ``one_sample_t_dz`` — repeated-measures pulse
  comparisons (cell-level within an experiment, and replicate-level across
  channels).
* ``mantel_test`` — distance-vs-correlation tests that respect the
  non-independence of cell *pairs*.
* ``population_permutation_pvalue`` — turns an existing per-cell permutation
  null into a population-level statement.
* ``bh_fdr`` / ``holm_correction`` — multiple-comparison correction.
* ``inferential_caveat`` — standardized figure footnote naming the unit of
  inference.

Pure functions: numpy/scipy in, plain dict/scalar out. No plotting, no state.
"""

import numpy as np
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    ttest_1samp,
    wilcoxon,
)


# =============================================================================
# Multiple-comparison correction
# =============================================================================
def bh_fdr(pvalues):
    """Benjamini-Hochberg adjusted p-values (q-values).

    ``pvalues`` is a 1-D array; non-finite entries (NaN) are excluded from the
    ranking and returned as NaN. Output is the same shape, clipped to ``<=1``.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pv = p[finite]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(n)
    res[order] = adj
    out[finite] = res
    return out


def holm_correction(pvalues):
    """Holm step-down adjusted p-values for a small post-hoc family.

    Same NaN handling as :func:`bh_fdr`. Holm is used for the (few) train-pair
    post-hoc contrasts where strong family-wise control is preferred.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pv = p[finite]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * (n - np.arange(n))
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(n)
    res[order] = adj
    out[finite] = res
    return out


# =============================================================================
# Repeated-measures pulse comparisons
# =============================================================================
def matched_pairs_rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation — Wilcoxon signed-rank effect size.

    Paired vectors ``x``, ``y``; NaN pairs and zero differences are dropped.
    Returns ``(W+ - W-) / (W+ + W-)`` in ``[-1, 1]`` on ``d = y - x``, so a
    positive value means ``y`` tends to exceed ``x``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(x) & ~np.isnan(y)
    d = y[valid] - x[valid]
    d = d[d != 0.0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    total = w_pos + w_neg
    if total == 0:
        return 0.0
    return float((w_pos - w_neg) / total)


def friedman_with_posthoc(matrix, *, correction="holm"):
    """Friedman repeated-measures omnibus + pairwise Wilcoxon post-hoc.

    ``matrix`` has shape ``(n_subjects, n_conditions)`` — rows are cells (or
    channels), columns are the repeated conditions (e.g. the 3 trains). Only
    complete-case rows (no NaN) are used.

    Returns a dict with ``friedman_stat`` / ``friedman_p``, ``kendall_w`` (the
    omnibus effect size ``chi2 / (n*(k-1))``) and ``posthoc`` — one entry per
    condition pair holding the Wilcoxon statistic, raw + adjusted p-values and
    the matched-pairs rank-biserial effect size. When fewer than 3 complete
    rows or 3 conditions are available, returns ``{"insufficient": True, ...}``.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return {"insufficient": True, "n": 0, "n_trains": 0}
    complete = arr[~np.isnan(arr).any(axis=1)]
    n, k = complete.shape
    if n < 3 or k < 3:
        return {"insufficient": True, "n": int(n), "n_trains": int(k)}

    stat, p = friedmanchisquare(*[complete[:, j] for j in range(k)])
    kendall_w = float(stat / (n * (k - 1))) if n * (k - 1) > 0 else np.nan

    posthoc, raw_ps = [], []
    for i in range(k):
        for j in range(i + 1, k):
            xi, yj = complete[:, i], complete[:, j]
            try:
                w_stat, w_p = wilcoxon(xi, yj, zero_method="wilcox")
            except ValueError:
                # All paired differences are zero — no signed-rank evidence.
                w_stat, w_p = np.nan, 1.0
            posthoc.append({
                "pair": (i, j),
                "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "p_raw": float(w_p),
                "rank_biserial": matched_pairs_rank_biserial(xi, yj),
            })
            raw_ps.append(w_p)

    adj = (holm_correction(np.array(raw_ps)) if correction == "holm"
           else bh_fdr(np.array(raw_ps)))
    for entry, a in zip(posthoc, adj):
        entry["p_adj"] = float(a)

    return {
        "insufficient": False,
        "n": int(n),
        "n_trains": int(k),
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "kendall_w": kendall_w,
        "posthoc": posthoc,
    }


def one_sample_t_dz(diffs):
    """One-sample t-test of per-replicate differences against 0, with Cohen's dz.

    ``diffs`` holds one value per biological replicate (e.g. each channel's
    ``train3 - train1`` mean response). NaN entries are dropped. Needs ``n>=2``;
    otherwise t / p / dz are NaN. ``cohen_dz = mean / sd``.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return {
            "n": int(n),
            "mean": float(d.mean()) if n else np.nan,
            "t_stat": np.nan, "p_value": np.nan, "cohen_dz": np.nan,
        }
    sd = d.std(ddof=1)
    res = ttest_1samp(d, 0.0)
    return {
        "n": int(n),
        "mean": float(d.mean()),
        "t_stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "cohen_dz": float(d.mean() / sd) if sd > 0 else np.nan,
    }


# =============================================================================
# Mantel test — distance vs correlation without pair pseudoreplication
# =============================================================================
def _safe_pearson(x, y):
    """Pearson r that returns NaN instead of raising on a constant input."""
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _within_group_perm(rng, groups):
    """Permutation of indices restricted to shuffle within each group label."""
    perm = np.arange(groups.size)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        perm[idx] = idx[rng.permutation(idx.size)]
    return perm


def mantel_test(dist_a, dist_b, *, n_perm=999, rng_seed=42, within_groups=None):
    """Mantel permutation test between two square symmetric matrices.

    ``dist_a`` and ``dist_b`` are ``(n_cells, n_cells)`` matrices (e.g. a
    pairwise distance matrix and a pairwise correlation matrix). The test
    correlates their off-diagonal entries, building a null by jointly
    permuting the row/column labels of ``dist_b`` — so the unit of
    exchangeability is the *cell*, not the pair, which fixes the
    pseudoreplication of treating ``n*(n-1)/2`` pairs as independent.

    Cells whose ``dist_b`` row carries non-finite entries (constant traces give
    NaN correlations) are dropped from both matrices before testing.
    ``within_groups`` (one integer label per cell) restricts permutations to
    within-group shuffles. Returns ``{"r_obs", "p_value", "n_perm", "n_cells"}``
    or ``{"insufficient": True, ...}`` when fewer than 4 usable cells remain.
    """
    A = np.asarray(dist_a, dtype=float)
    B = np.asarray(dist_b, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        return {"insufficient": True, "n_cells": 0}

    n0 = A.shape[0]
    off = ~np.eye(n0, dtype=bool)
    finite_cell = (
        np.isfinite(np.where(off, A, 0.0)).all(axis=1)
        & np.isfinite(np.where(off, B, 0.0)).all(axis=1)
    )
    if int(finite_cell.sum()) < 4:
        return {"insufficient": True, "n_cells": int(finite_cell.sum())}
    A = A[np.ix_(finite_cell, finite_cell)]
    B = B[np.ix_(finite_cell, finite_cell)]
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    a = A[iu]

    r_obs = _safe_pearson(a, B[iu])
    if not np.isfinite(r_obs):
        return {"insufficient": True, "n_cells": int(n)}

    groups = None
    if within_groups is not None:
        groups = np.asarray(within_groups)[finite_cell]

    rng = np.random.default_rng(rng_seed)
    count = 0
    for _ in range(n_perm):
        perm = (_within_group_perm(rng, groups) if groups is not None
                else rng.permutation(n))
        r_p = _safe_pearson(a, B[np.ix_(perm, perm)][iu])
        if np.isfinite(r_p) and abs(r_p) >= abs(r_obs):
            count += 1
    return {
        "insufficient": False,
        "r_obs": float(r_obs),
        "p_value": (1.0 + count) / (1.0 + n_perm),
        "n_perm": int(n_perm),
        "n_cells": int(n),
    }


# =============================================================================
# Population-level inference from an existing per-cell permutation null
# =============================================================================
def population_permutation_pvalue(observed_per_cell, null_mat, *, statistic="mean"):
    """Population-level one-tailed p-value from a per-cell permutation null.

    ``observed_per_cell`` has shape ``(n_cells,)`` and ``null_mat`` shape
    ``(n_perm, n_cells)`` — the same null already produced for per-cell
    p-values. The population statistic (``mean`` or ``median`` across cells) is
    compared to its per-permutation null distribution, answering "does the
    *population* score exceed chance" rather than "which individual cells do".
    Returns the observed statistic, the null mean/std, a z-like effect size and
    the one-tailed p (fraction of permutations at least as extreme).
    """
    obs = np.asarray(observed_per_cell, dtype=float)
    null = np.asarray(null_mat, dtype=float)
    if statistic == "mean":
        obs_stat = float(np.nanmean(obs))
        null_stats = np.nanmean(null, axis=1)
    elif statistic == "median":
        obs_stat = float(np.nanmedian(obs))
        null_stats = np.nanmedian(null, axis=1)
    else:
        raise ValueError(f"Unknown statistic={statistic!r}")
    n_perm = int(null_stats.size)
    p = (1.0 + float(np.sum(null_stats >= obs_stat))) / (1.0 + n_perm)
    null_mean = float(np.nanmean(null_stats))
    null_std = float(np.nanstd(null_stats, ddof=1)) if n_perm > 1 else np.nan
    z = ((obs_stat - null_mean) / null_std
         if np.isfinite(null_std) and null_std > 0 else np.nan)
    return {
        "obs_stat": obs_stat,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": float(z) if np.isfinite(z) else np.nan,
        "p_value": float(p),
        "n_perm": n_perm,
    }


# =============================================================================
# Figure footnote
# =============================================================================
def inferential_caveat(exp_name, n_channels, *, unit, extra=""):
    """Standardized one-line caveat naming the unit of inference for a figure.

    ``unit`` is the observational unit the plotted test uses (e.g. ``"cell"``);
    ``n_channels`` is the biological-replicate count. With a single channel the
    note flags the figure as within-experiment / descriptive only.
    """
    if n_channels >= 2:
        rep = (f"channels = biological replicates (n={n_channels}); "
               f"replicate-level test is the unit of biological inference")
    else:
        rep = ("n=1 biological replicate — within-experiment / descriptive "
               "only, no biological generalization")
    msg = f"Inferential unit: {unit}.  {rep}."
    if extra:
        msg += f"  {extra}"
    return msg
