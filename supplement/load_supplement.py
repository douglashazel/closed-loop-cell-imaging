#!/usr/bin/env python3
"""Rebuild the Stage-2 pipeline state from an exported supplement bundle.

``common.pipeline.prepare_state`` normally regenerates the background-corrected
fluorescence by re-reading every raw microscope frame (~100 GB, not
redistributed) to fit a per-frame 2-D polynomial background. This module skips
that step: it reads the already-corrected tables written by
``export_supplement.py`` and assembles a ``state`` dict that is numerically
identical to the one ``prepare_state`` produces, so the ``analyze_*.py``
scripts run unchanged.

Typical use, from the project root::

    # rebuild state and run the secondary analyses from the exported tables
    python SCRIPTS/preprint_analysis/load_supplement.py \
        --root supplement \
        --analyses responders dff clustering learning_scores

    # then render figures as usual
    ./run_aggregate_plots.sh

Or programmatically::

    import load_supplement
    experiments, state = load_supplement.load_state("supplement")

Analyses that additionally need the raw images (``responder_diagnostic``'s
frame-sharpness panel, and the frame mosaics) cannot be reproduced from this
bundle; everything else can.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import EXPERIMENTS

EXPORT_VERSION = 1

# Analyses reproducible from an exported bundle, in dependency order
# (``responders`` writes the shared mask other analyses read).
RUNNABLE = [
    "responders",
    "dff",
    "average_peak",
    "correlation_distance",
    "clustering",
    "response_violins",
    "learning_scores",
]

# Analyses that need the raw frames and therefore cannot run from the bundle.
NEEDS_RAW_FRAMES = ["responder_diagnostic", "mosaics"]


# =============================================================================
# state reconstruction
# =============================================================================
# pandas' default CSV reader uses a fast but not correctly-rounded float
# parser, which costs ~1 ULP per value. The tables are written with Python's
# shortest round-trip repr, so reading with float_precision="round_trip"
# recovers the exact float64 bits the pipeline produced.
_READ = {"float_precision": "round_trip"}


def _read_matrix_csv(path):
    """Read a ``cell_id`` + ``f<N>`` matrix CSV into (ids, frame_nums, values)."""
    df = pd.read_csv(path, **_READ)
    cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    frame_nums = np.array([int(str(c).lstrip("f")) for c in cols])
    ids = [int(v) for v in df["cell_id"].values]
    return ids, frame_nums, df[cols].values.astype(np.float64)


def _load_chamber(root, entry):
    """Load one chamber's exported files into plain Python structures."""
    chamber = entry["chamber"]
    d = os.path.join(root, chamber)

    with open(os.path.join(d, f"{chamber}_metadata.json")) as f:
        meta = json.load(f)
    if meta.get("EXPORT_VERSION") != EXPORT_VERSION:
        raise RuntimeError(
            f"{chamber}: bundle EXPORT_VERSION={meta.get('EXPORT_VERSION')}, "
            f"this loader expects {EXPORT_VERSION}"
        )

    ids, frame_nums, mat = _read_matrix_csv(
        os.path.join(d, f"{chamber}_fluorescence_bgcorrected.csv")
    )
    # Cell IDs are strings in the pipeline's dicts (they originate as JSON
    # object keys); keep that so downstream key lookups behave identically.
    corrected = {
        str(cid): {f"f{int(n)}": float(v) for n, v in zip(frame_nums, row)}
        for cid, row in zip(ids, mat)
    }

    pos = pd.read_csv(os.path.join(d, f"{chamber}_cell_positions.csv"), **_READ)
    traj = {}
    for cid, sub in pos.groupby("cell_id", sort=False):
        coords = {}
        for fr, x, y in zip(sub["frame"].values, sub["x"].values, sub["y"].values):
            coords[f"x{int(fr)}"] = float(x)
            coords[f"y{int(fr)}"] = float(y)
        traj[str(int(cid))] = coords

    bg = pd.read_csv(os.path.join(d, f"{chamber}_background.csv"), **_READ)
    n_analyzed = int(meta["n_frames_analyzed"])
    # bg_trace is clipped to the analysis window (as prepare_state leaves it);
    # bg_fit_min stays at full recording length.
    bg_trace = bg["bg_sampled_mean"].values[:n_analyzed].astype(np.float32)
    bg_min = bg["bg_fit_min"].values.astype(np.float64)

    src = meta["frame_minutes_src"]
    frame_minutes_src = (
        np.asarray(src["frames"], dtype=float),
        np.asarray(src["minutes"], dtype=float),
    )

    return meta, corrected, traj, bg_trace, bg_min, frame_minutes_src


def load_state(root="supplement", experiments=None):
    """Return ``(experiments_cfg, state)`` rebuilt from the bundle at ``root``.

    ``experiments_cfg`` is a deep copy of ``common.config.EXPERIMENTS`` holding
    only the experiments present in the bundle, with ``stim_frames`` filled in
    from the exported metadata (normally done by ``resolve_all_stim_frames``,
    which needs the hardware logs / timestamp CSVs).
    """
    with open(os.path.join(root, "index.json")) as f:
        index = json.load(f)

    wanted = set(experiments) if experiments else None
    entries = [
        e for e in index["chambers"]
        if wanted is None or e["experiment"] in wanted
    ]
    if not entries:
        raise RuntimeError(f"No chambers in {root} match {experiments}")

    exp_names = []
    for e in entries:
        if e["experiment"] not in exp_names:
            exp_names.append(e["experiment"])
    cfgs = {name: copy.deepcopy(EXPERIMENTS[name]) for name in exp_names}

    state = {
        "corrected_lum": {n: {} for n in exp_names},
        "traj_by_channel": {n: {} for n in exp_names},
        "frame_counts": {n: {} for n in exp_names},
        "bg_trace": {n: {} for n in exp_names},
        "bg_min_by_ch": {n: {} for n in exp_names},
        "frame_minutes_src": {n: {} for n in exp_names},
        "real_setpoint_min": {n: {} for n in exp_names},
    }
    for name in exp_names:
        cfgs[name]["stim_frames"] = {}

    loaded = {n: [] for n in exp_names}
    for e in entries:
        meta, corrected, traj, bg_trace, bg_min, fms = _load_chamber(root, e)
        exp, ch = meta["experiment"], meta["channel"]
        state["corrected_lum"][exp][ch] = corrected
        state["traj_by_channel"][exp][ch] = traj
        state["frame_counts"][exp][ch] = int(meta["n_frames_analyzed"])
        state["bg_trace"][exp][ch] = bg_trace
        state["bg_min_by_ch"][exp][ch] = bg_min
        state["frame_minutes_src"][exp][ch] = fms
        state["real_setpoint_min"][exp][ch] = meta["real_setpoint_min"]
        cfgs[exp]["stim_frames"][ch] = [int(f) for f in meta["stim_frames"]]
        loaded[exp].append(ch)

    # Keep channel order as declared in config, restricted to what was loaded.
    for name in exp_names:
        cfgs[name]["channels"] = [
            c for c in EXPERIMENTS[name]["channels"] if c in loaded[name]
        ]

    return cfgs, state


# =============================================================================
# running the analyses off the bundle
# =============================================================================
def install(root="supplement"):
    """Monkeypatch ``prepare_state`` so ``analyze_*.py`` read the bundle.

    Patches ``common.pipeline`` and any ``analyze_*`` module already imported
    (they do ``from common.pipeline import prepare_state``, which binds a local
    name at import time).
    """
    import common.pipeline as pipeline

    def _patched(experiments, *, recompute_bg=False, check_direction=True):
        _, state = load_state(root, experiments=list(experiments))
        return state

    pipeline.prepare_state = _patched
    for name, mod in list(sys.modules.items()):
        if name.startswith("analyze_") and hasattr(mod, "prepare_state"):
            mod.prepare_state = _patched
    return _patched


def _run(analyses, root, cache_dir):
    import common.io_paths as io_paths

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        io_paths.ANALYSIS_CACHE_DIR = cache_dir
        print(f"analysis cache -> {cache_dir}")

    experiments, state = load_state(root)
    print(
        f"Loaded {sum(len(c['channels']) for c in experiments.values())} chambers "
        f"from {root}: {list(experiments)}"
    )

    import importlib

    for a in analyses:
        if a in NEEDS_RAW_FRAMES:
            print(f"  SKIP {a}: needs the raw image frames (not redistributed)")
            continue
        mod = importlib.import_module(f"analyze_{a}")
        fn = getattr(mod, "analyze", None) or getattr(mod, "compute_and_cache")
        print(f">>> {a}")
        fn(experiments, state)
    print("\nDone. Render with ./run_aggregate_plots.sh")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default="supplement")
    ap.add_argument(
        "--analyses", nargs="+", default=RUNNABLE,
        help=f"subset of: {' '.join(RUNNABLE)}",
    )
    ap.add_argument(
        "--analysis-cache-dir", default=None,
        help="override results/analysis_cache (useful for side-by-side checks)",
    )
    args = ap.parse_args()
    _run(args.analyses, args.root, args.analysis_cache_dir)


if __name__ == "__main__":
    main()
