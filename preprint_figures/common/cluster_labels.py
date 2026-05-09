"""Persist + load PCA/KMeans cluster labels so downstream scripts can stratify.

clustering.py writes one pickle per (experiment, channel-or-pooled-key):
    CACHE_DIR/clusters/<exp>__<ch_key>.pkl

The pickle contains the cell IDs in the same row order as the labels so any
downstream script can re-align labels to its own cell ordering. ``ch_key`` is
either a single channel name or the literal ``"__pooled__"`` for the across-
channel clustering result.

Cell IDs are stored as ``[(ch, cid)]`` tuples even for single-channel runs,
which makes per-channel and pooled caches consume the same alignment helper.
"""

import os
import pickle
from typing import Iterable

import numpy as np

from common.config import CACHE_DIR


CLUSTER_CACHE_DIR = os.path.join(CACHE_DIR, "clusters")
os.makedirs(CLUSTER_CACHE_DIR, exist_ok=True)


def _cluster_cache_path(exp_name, ch_key):
    safe = str(ch_key).replace("/", "_")
    return os.path.join(CLUSTER_CACHE_DIR, f"{exp_name}__{safe}.pkl")


def save_cluster_labels(
    exp_name, ch_key, *, cell_ids, labels, best_k, **extra,
):
    """Pickle cluster labels keyed by ``(exp_name, ch_key)``.

    ``cell_ids`` and ``labels`` must be row-aligned. ``cell_ids`` is a list of
    ``(channel, cid)`` tuples. ``best_k`` is the silhouette-chosen k.
    """
    blob = {
        "cell_ids": list(cell_ids),
        "labels": np.asarray(labels, dtype=np.int64),
        "best_k": int(best_k),
        **extra,
    }
    path = _cluster_cache_path(exp_name, ch_key)
    with open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cluster_labels(exp_name, ch_key):
    """Return the cached blob or None if missing."""
    path = _cluster_cache_path(exp_name, ch_key)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def align_labels_to_cells(blob, target_cell_ids: Iterable):
    """Return an int array of cluster labels aligned to ``target_cell_ids``.

    Cells not present in the cache get label ``-1`` so callers can drop them.
    ``target_cell_ids`` is an iterable of ``(channel, cid)`` tuples that match
    the format used at save time.
    """
    if blob is None:
        target = list(target_cell_ids)
        return np.full(len(target), -1, dtype=np.int64)
    lookup = {tuple(k): int(v) for k, v in zip(blob["cell_ids"], blob["labels"])}
    out = np.array(
        [lookup.get(tuple(t), -1) for t in target_cell_ids],
        dtype=np.int64,
    )
    return out


def iter_clusters(blob_or_target, target_cell_ids=None):
    """Yield ``(cluster_tag, mask_or_None)`` over an "all + per-cluster" sequence.

    Two call patterns:
        * ``iter_clusters(None, target_cell_ids)`` → yields only ``("all", None)``.
        * ``iter_clusters(blob, target_cell_ids)`` → yields ``("all", None)``
          first, then one ``(f"c{cid}", mask)`` per cluster.

    ``mask`` is a boolean array of length ``len(target_cell_ids)``.
    Clusters with fewer than 3 matching cells are skipped (not enough for a
    meaningful split).
    """
    yield ("all", None)
    if blob_or_target is None:
        return
    aligned = align_labels_to_cells(blob_or_target, target_cell_ids)
    for cid in range(int(blob_or_target["best_k"])):
        mask = aligned == cid
        if int(mask.sum()) < 3:
            continue
        yield (f"c{cid}", mask)
