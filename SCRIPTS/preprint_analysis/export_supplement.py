#!/usr/bin/env python3
"""Export per-chamber supplementary data tables for the preprint.

Writes, for each of the 8 CellASIC chambers featured in the paper, a
self-contained bundle of CSV/NPZ/JSON files under
``results/supplement_export/<CHAMBER>/``:

    <CHAMBER>_fluorescence_raw.csv          all segmented cells x all frames,
                                            uncorrected ROI mean intensity
    <CHAMBER>_fluorescence_bgcorrected.csv  analysed cells x analysed frames,
                                            after polynomial background
                                            subtraction (+ dead-frame fill)
    <CHAMBER>_dff.csv                       (F - F0) / F0 of the above
    <CHAMBER>_cell_positions.csv            tidy cell_id/frame/x/y tracks
    <CHAMBER>_background.csv                per-frame background scalars
    <CHAMBER>_time_axis.csv                 per-frame minutes + stim flag
    <CHAMBER>_mask.npz / .png               frame-0 Cellpose label mask
    <CHAMBER>_mask_preview.png              colour rendering of the above
    <CHAMBER>_mask_analyzed_cells.npz/.png  cell-selection mask (C2C12 only)
    <CHAMBER>_metadata.json                 everything else needed to rerun

Why the background-corrected table matters: ``common.pipeline.prepare_state``
regenerates normalisation by re-reading every raw frame to fit a per-frame 2-D
polynomial background. The raw frames (~100 GB) are not redistributed, so the
corrected table is the bridge that lets a third party rerun Stage 2 without
them -- see ``load_supplement.py``.

This script is read-only with respect to every existing pipeline output: it
reads the warm ``results/bg_cache/`` pickles and writes only under
``results/supplement_export/``.

Run from the project root:

    python SCRIPTS/preprint_analysis/export_supplement.py
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import EXPERIMENTS, cell_line_label
from common.io_paths import channel_dir, load_segmentation
from common.pipeline import prepare_state, resolve_dead_frame_indices
from common.stim_helpers import compute_f0_baseline
from common.time_axis import frames_to_min, response_window_frames

sys.path.insert(0, "SCRIPTS/core_pipeline")
from io_utils import load_msgpack, lum_dict_to_df  # noqa: E402


DEFAULT_OUT = os.path.join("results", "supplement_export")

# Public chamber name for each (experiment, internal channel). The letters are
# the chamber labels used in the manuscript; the internal channel strings are
# the keys in common.config.EXPERIMENTS.
CHAMBERS = [
    ("c2c12_dmso_09APR26", "channel 1", "C2C12_A"),
    ("c2c12_dmso_09APR26", "channel 2", "C2C12_B"),
    ("c2c12_dmso_09APR26", "channel 3", "C2C12_C"),
    ("pc3_dmso_23MAR26", "channel 1", "PC3"),
    ("nrk_acid_13APR26", "channel 1 A", "NRK_A"),
    ("nrk_acid_13APR26", "channel 2 B", "NRK_B"),
    ("nrk_acid_13APR26", "channel 1 C", "NRK_C"),
    ("nrk_acid_13APR26", "channel 2 D", "NRK_D"),
]

# Format version for the export layout; load_supplement.py checks it.
EXPORT_VERSION = 1


# =============================================================================
# helpers
# =============================================================================
def _frame_cols(df):
    """Sorted 'f<N>' columns of ``df`` and their integer frame indices."""
    cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    return cols, np.array([int(str(c).lstrip("f")) for c in cols])


def _write_matrix_csv(path, cell_ids, frame_nums, mat):
    """Write a cells x frames matrix with a ``cell_id`` index column.

    Values go out through pandas' default float formatting, which uses Python's
    shortest round-trip ``repr`` -- so float64 reloads bit-identically.
    """
    df = pd.DataFrame(mat, columns=[f"f{int(n)}" for n in frame_nums])
    df.insert(0, "cell_id", list(cell_ids))
    df.to_csv(path, index=False)
    return df


def _mask_files(cdir):
    """Sorted ``.npy`` mask filenames in ``<cdir>/masks``."""
    mdir = os.path.join(cdir, "masks")
    if not os.path.isdir(mdir):
        return mdir, []
    return mdir, sorted(f for f in os.listdir(mdir) if f.endswith(".npy"))


def _label_colors(n_labels):
    """Deterministic RGB lookup table for labels 0..n_labels (0 = black).

    Hue advances by the golden ratio so consecutive labels -- which are usually
    neighbouring cells -- never land on similar colours.
    """
    idx = np.arange(n_labels + 1, dtype=np.float64)
    hsv = np.stack([
        (idx * 0.6180339887) % 1.0,
        np.where(idx % 2, 0.70, 1.00),
        np.where(idx % 3, 1.00, 0.75),
    ], axis=1)
    lut = (hsv_to_rgb(hsv) * 255).round().astype(np.uint8)
    lut[0] = 0
    return lut


def _save_mask_preview(m16, png_path):
    """Write an 8-bit colour rendering of a label mask.

    The published masks are 16-bit label images: pixel value = cell ID, so the
    brightest cell in a 500-cell chamber sits at 500/65535 and the file looks
    black in an ordinary image viewer. This preview exists purely so the
    segmentation can be eyeballed; it is not analysis input.
    """
    Image.fromarray(_label_colors(int(m16.max()))[m16], mode="RGB").save(
        png_path, optimize=True,
    )


def _save_mask(mask, npz_path, png_path, preview_path):
    """Save a label mask as compressed NPZ, 16-bit PNG, and colour preview."""
    m = np.asarray(mask)
    if m.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"mask has {m.max()} labels — exceeds uint16")
    m16 = m.astype(np.uint16)
    np.savez_compressed(npz_path, masks=m16)
    Image.fromarray(m16, mode="I;16").save(png_path, optimize=True)
    _save_mask_preview(m16, preview_path)
    return m16


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# per-chamber export
# =============================================================================
def export_chamber(state, exp_name, ch, chamber, out_root):
    cfg = EXPERIMENTS[exp_name]
    cdir = channel_dir(cfg, ch)
    out_dir = os.path.join(out_root, chamber)
    os.makedirs(out_dir, exist_ok=True)

    def out(suffix):
        return os.path.join(out_dir, f"{chamber}_{suffix}")

    # ---- corrected + dF/F0 (exactly what the analyses consume) -------------
    corr = state["corrected_lum"][exp_name][ch]
    cdf = lum_dict_to_df(corr).set_index("CellID")
    cols, frame_nums = _frame_cols(cdf)
    mat = cdf[cols].values
    analyzed_ids = [int(i) for i in cdf.index]

    F0, baseline_cols, first_stim = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff = (mat - F0) / F0_safe

    _write_matrix_csv(out("fluorescence_bgcorrected.csv"), analyzed_ids, frame_nums, mat)
    _write_matrix_csv(out("dff.csv"), analyzed_ids, frame_nums, dff)

    # ---- raw (all segmented+tracked cells, full recording length) ----------
    raw = load_msgpack(os.path.join(cdir, "analysis", "luminosity_complete.json"))
    rdf = lum_dict_to_df(raw).set_index("CellID")
    rcols, rframe_nums = _frame_cols(rdf)
    _write_matrix_csv(
        out("fluorescence_raw.csv"), [int(i) for i in rdf.index], rframe_nums,
        rdf[rcols].values,
    )
    n_frames_total = int(len(rframe_nums))

    # ---- cell positions (tidy; analysed cells, full track length) ----------
    traj = state["traj_by_channel"][exp_name][ch]
    rows = []
    for cid, coords in traj.items():
        cid_i = int(cid)
        for key, val in coords.items():
            if not key.startswith("x"):
                continue
            i = key[1:]
            if not i.isdigit():
                continue
            y = coords.get(f"y{i}")
            if val is None or y is None:
                continue
            rows.append((cid_i, int(i), float(val), float(y)))
    pos = pd.DataFrame(rows, columns=["cell_id", "frame", "x", "y"])
    pos = pos.sort_values(["cell_id", "frame"]).reset_index(drop=True)
    pos.to_csv(out("cell_positions.csv"), index=False)

    # ---- per-frame background scalars (full length; bg_trace is pre-clip) --
    bg_trace_clipped = np.asarray(state["bg_trace"][exp_name][ch])
    bg_min_full = np.asarray(state["bg_min_by_ch"][exp_name][ch], dtype=np.float64)
    n_bg = int(len(bg_min_full))
    bg_mean_full = np.full(n_bg, np.nan, dtype=np.float32)
    bg_mean_full[: len(bg_trace_clipped)] = bg_trace_clipped.astype(np.float32)
    pd.DataFrame({
        "frame": np.arange(n_bg, dtype=int),
        "bg_sampled_mean": bg_mean_full,
        "bg_fit_min": bg_min_full,
    }).to_csv(out("background.csv"), index=False)

    # ---- time axis --------------------------------------------------------
    known_frames, known_minutes = state["frame_minutes_src"][exp_name][ch]
    all_frames = np.arange(n_frames_total, dtype=float)
    all_minutes = frames_to_min(state, exp_name, ch, all_frames)
    stim_frames = [int(f) for f in cfg["stim_frames"][ch]]
    stim_set = set(stim_frames)
    pd.DataFrame({
        "frame": all_frames.astype(int),
        "minutes": all_minutes,
        "is_stim_onset": [int(f) in stim_set for f in all_frames.astype(int)],
        "in_analysis_window": all_frames.astype(int) < len(frame_nums),
    }).to_csv(out("time_axis.csv"), index=False)

    # ---- masks ------------------------------------------------------------
    mask_info = {}
    mdir, mfiles = _mask_files(cdir)
    if mfiles:
        m0 = load_segmentation(os.path.join(mdir, mfiles[0]))
        m16 = _save_mask(
            m0, out("mask.npz"), out("mask.png"), out("mask_preview.png"),
        )
        mask_info["frame0_mask_source"] = mfiles[0]
        mask_info["frame0_mask_shape"] = list(m16.shape)
        mask_info["frame0_mask_n_labels"] = int(len(np.unique(m16)) - 1)
        mask_info["n_mask_frames_available"] = len(mfiles)
    else:
        mask_info["frame0_mask_source"] = None

    filt_rel = (cfg.get("cell_mask_filter") or {}).get(ch)
    if filt_rel:
        fm = load_segmentation(os.path.join(cfg["dir"], filt_rel))
        fm16 = _save_mask(
            fm, out("mask_analyzed_cells.npz"), out("mask_analyzed_cells.png"),
            out("mask_analyzed_cells_preview.png"),
        )
        mask_info["analyzed_cells_mask_source"] = os.path.basename(filt_rel)
        mask_info["analyzed_cells_mask_n_labels"] = int(len(np.unique(fm16)) - 1)

    # ---- closed-loop hardware feedback log (NRK only) ---------------------
    # The sibling monitoring.log is deliberately NOT redistributed: it is
    # ~17 MB of pipeline chatter containing internal absolute filesystem
    # paths. luminosity_log_channel<N>.json holds the actual feedback record
    # (per-frame mean luminosity, setpoint, and the controller's decision).
    log_spec = (cfg.get("stim_logs") or {}).get(ch)
    if log_spec:
        log_path, ch_num = log_spec
        lum_log = os.path.join(
            os.path.dirname(log_path), f"luminosity_log_channel{ch_num}.json",
        )
        if os.path.exists(lum_log):
            with open(lum_log) as f:
                entries = json.load(f)
            entries = [e for e in entries if e.get("channel") == ch_num]
            entries.sort(key=lambda e: e["frame"])
            with open(out("hardware_feedback_log.json"), "w") as f:
                json.dump(entries, f, indent=1)
            mask_info["hardware_feedback_log_entries"] = len(entries)

    # ---- dead frames ------------------------------------------------------
    if cfg.get("filter_dead_frames"):
        dead = resolve_dead_frame_indices(cfg, bg_trace_clipped)
        dead_frames = [int(d) for d in dead]
    else:
        dead_frames = []

    # ---- metadata ---------------------------------------------------------
    lo, hi = response_window_frames(state, exp_name, ch, cfg)
    meta = {
        "EXPORT_VERSION": EXPORT_VERSION,
        "chamber": chamber,
        "experiment": exp_name,
        "channel": ch,
        "cell_line": cell_line_label(exp_name),
        "source_dir": cdir,

        "n_frames_total": n_frames_total,
        "n_frames_analyzed": int(len(frame_nums)),
        "n_cells_raw": int(rdf.shape[0]),
        "n_cells_analyzed": int(len(analyzed_ids)),
        "analyzed_cell_ids": analyzed_ids,

        "stim_frames": stim_frames,
        "stim_minutes": [float(m) for m in frames_to_min(
            state, exp_name, ch, np.asarray(stim_frames, dtype=float),
        )] if stim_frames else [],
        "stim_label": cfg.get("stim_label"),
        "stim_duration_minutes": cfg.get("stim_duration_minutes"),
        "response_direction": cfg.get("response_direction"),
        "response_window_frames": [int(lo), int(hi)],
        "response_window_minutes": list(cfg["response_window_minutes"])
            if cfg.get("response_window_minutes") else None,

        "f0_first_stim_frame": int(first_stim),
        "f0_baseline_frames": [int(str(c).lstrip("f")) for c in baseline_cols],

        "time_window_minutes": cfg.get("time_window_minutes"),
        "filter_dead_frames": bool(cfg.get("filter_dead_frames")),
        "dead_frames": dead_frames,
        "real_setpoint_min": state["real_setpoint_min"][exp_name].get(ch),

        # Interpolation source for frame -> minutes. Reproduces frames_to_min
        # exactly (np.interp over these knots); for NRK these come from the
        # hardware monitoring.log and are NOT one-per-frame.
        "frame_minutes_src": {
            "frames": [float(v) for v in np.asarray(known_frames)],
            "minutes": [float(v) for v in np.asarray(known_minutes)],
        },
    }
    meta.update(mask_info)
    with open(out("metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"  {chamber:9s} cells raw={meta['n_cells_raw']:4d} "
        f"analysed={meta['n_cells_analyzed']:4d} | frames "
        f"total={n_frames_total:4d} analysed={meta['n_frames_analyzed']:4d} | "
        f"stims={len(stim_frames)}"
    )
    return meta


# =============================================================================
# driver
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root directory")
    args = ap.parse_args()

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    print("Building pipeline state (reads warm results/bg_cache/ pickles)...")
    state = prepare_state(EXPERIMENTS, recompute_bg=False, check_direction=False)

    print("\nExporting chambers:")
    metas = []
    for exp_name, ch, chamber in CHAMBERS:
        metas.append(export_chamber(state, exp_name, ch, chamber, out_root))

    # Ship the loader and the README next to the data so the bundle stands alone.
    here = os.path.dirname(os.path.abspath(__file__))
    for src, dst in [
        ("load_supplement.py", "load_supplement.py"),
        ("supplement_README.md", "README.md"),
    ]:
        p = os.path.join(here, src)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_root, dst))

    index = {
        "EXPORT_VERSION": EXPORT_VERSION,
        "chambers": [
            {k: m[k] for k in
             ("chamber", "experiment", "channel", "cell_line",
              "n_cells_analyzed", "n_frames_analyzed")}
            for m in metas
        ],
    }
    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Checksums over everything written.
    lines = []
    for dirpath, _, files in os.walk(out_root):
        for fn in sorted(files):
            if fn == "CHECKSUMS.sha256":
                continue
            p = os.path.join(dirpath, fn)
            lines.append(f"{_sha256(p)}  {os.path.relpath(p, out_root)}")
    with open(os.path.join(out_root, "CHECKSUMS.sha256"), "w") as f:
        f.write("\n".join(sorted(lines, key=lambda s: s.split("  ", 1)[1])) + "\n")

    total = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fs in os.walk(out_root) for fn in fs
    )
    print(f"\nWrote {len(metas)} chambers to {out_root}/  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
