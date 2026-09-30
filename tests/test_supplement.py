"""Consistency checks on the published supplement bundle (supplement/)."""
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "supplement")
READ = {"float_precision": "round_trip"}  # exact float64, as load_supplement.py reads

with open(os.path.join(BUNDLE, "index.json")) as _f:
    INDEX = json.load(_f)
CHAMBERS = [e["chamber"] for e in INDEX["chambers"]]

# Responders per chamber as published
PUBLISHED_RESPONDERS = {
    "C2C12_A": 14, "C2C12_B": 38, "C2C12_C": 23, "PC3": 136,
    "NRK_A": 28, "NRK_B": 57, "NRK_C": 33, "NRK_D": 41,
}


def _path(chamber, suffix):
    return os.path.join(BUNDLE, chamber, f"{chamber}_{suffix}")


def _meta(chamber):
    with open(_path(chamber, "metadata.json")) as f:
        return json.load(f)


def _matrix(chamber, name):
    df = pd.read_csv(_path(chamber, name), index_col="cell_id", **READ)
    return df


def _readme_table():
    """{chamber: (tracked, analysed, frames)} from the README chamber table."""
    rows = {}
    with open(os.path.join(BUNDLE, "README.md"), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\| `(\w+)` \|.*\| (\d+) \| (\d+) \| (\d+) \|\s*$", line)
            if m:
                rows[m.group(1)] = tuple(int(g) for g in m.groups()[1:])
    return rows


def test_checksums():
    with open(os.path.join(BUNDLE, "CHECKSUMS.sha256")) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    listed = set()
    for line in lines:
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        with open(os.path.join(BUNDLE, rel), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == digest, rel
    on_disk = {
        os.path.relpath(os.path.join(dp, fn), BUNDLE)
        for dp, dirs, fns in os.walk(BUNDLE)
        for fn in fns
        if fn != "CHECKSUMS.sha256" and "__pycache__" not in dp
    }
    assert on_disk == listed


@pytest.mark.parametrize("chamber", CHAMBERS)
def test_counts_agree(chamber):
    meta = _meta(chamber)
    entry = next(e for e in INDEX["chambers"] if e["chamber"] == chamber)
    raw = _matrix(chamber, "fluorescence_raw.csv")
    corr = _matrix(chamber, "fluorescence_bgcorrected.csv")
    dff = _matrix(chamber, "dff.csv")

    assert entry["n_cells_analyzed"] == meta["n_cells_analyzed"] == len(corr) == len(dff)
    assert entry["n_frames_analyzed"] == meta["n_frames_analyzed"] == corr.shape[1] == dff.shape[1]
    assert meta["n_cells_raw"] == len(raw)
    assert meta["n_frames_total"] == raw.shape[1]
    assert list(corr.index) == list(dff.index) == meta["analyzed_cell_ids"]
    assert _readme_table()[chamber] == (len(raw), len(corr), corr.shape[1])


@pytest.mark.parametrize("chamber", CHAMBERS)
def test_positions_cover_analysed_cells(chamber):
    pos = pd.read_csv(_path(chamber, "cell_positions.csv"), **READ)
    assert set(pos["cell_id"]) == set(_meta(chamber)["analyzed_cell_ids"])


@pytest.mark.parametrize("chamber", CHAMBERS)
def test_stim_onsets(chamber):
    meta = _meta(chamber)
    t = pd.read_csv(_path(chamber, "time_axis.csv"), **READ)
    assert len(t) == meta["n_frames_total"]
    assert list(t.loc[t["is_stim_onset"], "frame"]) == sorted(meta["stim_frames"])
    assert int(t["in_analysis_window"].sum()) == meta["n_frames_analyzed"]


@pytest.mark.parametrize("chamber", CHAMBERS)
def test_masks_npz_png_identical(chamber):
    suffixes = ["mask"]
    if os.path.exists(_path(chamber, "mask_analyzed_cells.npz")):
        suffixes.append("mask_analyzed_cells")
    for s in suffixes:
        with np.load(_path(chamber, f"{s}.npz")) as z:
            npz = z["masks"]
        png = np.asarray(Image.open(_path(chamber, f"{s}.png")))
        assert npz.dtype == np.uint16
        assert np.array_equal(npz, png.astype(np.uint16)), s


@pytest.mark.parametrize("chamber", CHAMBERS)
def test_dff_recomputes_exactly(chamber):
    meta = _meta(chamber)
    corr = _matrix(chamber, "fluorescence_bgcorrected.csv")
    dff = _matrix(chamber, "dff.csv")
    base = [f"f{i}" for i in meta["f0_baseline_frames"]]
    mat = corr.values
    f0 = np.nanmean(corr[base].values, axis=1, keepdims=True)
    f0_safe = np.where(f0 == 0, np.nan, f0)
    expected = (mat - f0) / f0_safe
    assert np.array_equal(expected, dff.values, equal_nan=True)


@pytest.mark.parametrize("chamber", CHAMBERS)
def test_no_duplicate_cells(chamber):
    raw = _matrix(chamber, "fluorescence_raw.csv")
    assert not raw.round(12).duplicated().any(), "two cells have identical raw traces"

    pos = pd.read_csv(_path(chamber, "cell_positions.csv"), **READ)
    n_frames = pos.groupby("cell_id")["frame"].nunique()
    shared = (
        pos.merge(pos, on=["frame", "x", "y"])
        .query("cell_id_x < cell_id_y")
        .groupby(["cell_id_x", "cell_id_y"]).size()
    )
    for (a, b), n in shared.items():
        assert n < 0.5 * min(n_frames[a], n_frames[b]), (a, b, n)


def test_responder_counts_from_bundle(tmp_path):
    """All experiments in one call must give the published responder counts."""
    cache = tmp_path / "cache"
    subprocess.run(
        [sys.executable, "SCRIPTS/preprint_analysis/load_supplement.py",
         "--analyses", "responders", "dff", "--analysis-cache-dir", str(cache)],
        cwd=ROOT, check=True, capture_output=True,
    )
    got = {}
    for e in INDEX["chambers"]:
        with open(cache / e["experiment"] / "responders.pkl", "rb") as f:
            blob = pickle.load(f)
        got[e["chamber"]] = blob["meta"]["n_responders"][e["channel"]]
    assert got == PUBLISHED_RESPONDERS
