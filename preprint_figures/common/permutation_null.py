"""Permutation-null helpers for learning scores.

The summed habituation / sensitization / anticipation score for a cell is a
function of the per-stim response array. Shuffling the stim ordering breaks
the train structure; comparing observed scores to the resulting null tells us
whether the observed score is concentrated more strongly than expected by
chance under random stim ordering.
"""

import numpy as np


def shuffle_per_stim_within_cell(rng, per_stim):
    """Independently permute stim order within each cell column.

    Shape preserved: ``(n_stims, n_cells)``. Right null for habituation /
    sensitization where the score depends on each cell's own running extremum
    across its own per-stim values.
    """
    out = per_stim.copy()
    n_stims, n_cells = out.shape
    for c in range(n_cells):
        out[:, c] = out[rng.permutation(n_stims), c]
    return out


def shuffle_per_stim(rng, per_stim):
    """Permute the stim axis globally (same permutation across all cells)."""
    perm = rng.permutation(per_stim.shape[0])
    return per_stim[perm]


def permutation_null_distribution(
    score_fn, per_stim, *, n_perm=200, rng_seed=42, mode="per_cell",
):
    """Run ``score_fn(shuffled_per_stim)`` for ``n_perm`` shuffles.

    Returns a ``(n_perm, n_cells)`` matrix of scores under the null. ``mode``
    selects ``per_cell`` (independent per-column shuffle) or ``global``
    (single permutation across the stim axis).
    """
    rng = np.random.default_rng(rng_seed)
    if mode == "per_cell":
        shuffler = shuffle_per_stim_within_cell
    elif mode == "global":
        shuffler = shuffle_per_stim
    else:
        raise ValueError(f"Unknown mode={mode!r}")
    n_cells = per_stim.shape[1]
    out = np.zeros((n_perm, n_cells), dtype=float)
    for i in range(n_perm):
        out[i] = score_fn(shuffler(rng, per_stim))
    return out


def pvalue_one_tailed(observed, null_mat):
    """One-tailed permutation p-value for "observed at least this large".

    ``observed`` shape ``(n_cells,)``; ``null_mat`` shape ``(n_perm, n_cells)``.
    Returns ``(1 + sum(null >= observed)) / (1 + n_perm)`` per cell.
    """
    n_perm = null_mat.shape[0]
    return (1.0 + np.sum(null_mat >= observed[None, :], axis=0)) / (1.0 + n_perm)
