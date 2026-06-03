#!/usr/bin/env python3
"""Average-peak analysis (no plotting) → analysis_cache/<exp>/average_peak.pkl.

DMSO experiments only. For each DMSO experiment every per-stimulus response
segment — the dF/F0 trace from a stimulus onset to ``SEGMENT_MINUTES`` (10 min)
later, the DMSO inter-stimulus interval — is pooled across all cells and all
channels and resampled onto a common 0-10 min grid. This step writes only the
RAW stacked segments the average-peak figures display:

  * ``all_segments_stacked``  — every cell×stim segment (all cells)
  * ``resp_segments_stacked`` — responder cells only (or None)
  * ``stim8_stacked``         — responder segments for stimulus #STIM8_INDEX
                                (or None)

The low-coverage tail clip (``MIN_COVERAGE_FRAC``) and the mean ± SEM are NOT
applied here — they are render-side concerns in ``plots/average_peak.py``. The
dF/F0 math is the verbatim ``(mat - F0) / F0_safe`` with F0 from
``compute_f0_baseline`` — identical to the original ``average_peak.py``. No
matplotlib / style imports.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_responders import get_responder_masks
from common.cli import parse_args
from common.config import cell_line_label
from common.io_paths import save_analysis_cache
from common.pipeline import prepare_state
from common.stim_helpers import compute_f0_baseline
from common.time_axis import frames_to_min

sys.path.insert(0, "SCRIPTS/core_pipeline")
from io_utils import lum_dict_to_df  # noqa: E402


# Window grabbed after each stimulus onset — the DMSO inter-stimulus interval
# (stims within a train are 10 min apart, so this is "until the next peak").
SEGMENT_MINUTES = 10.0
# Common grid the per-stimulus segments are resampled onto. Channels / experi-
# ments have different frame rates, so segments hold different frame counts;
# np.interp puts them all on the same 0-SEGMENT_MINUTES axis.
GRID_POINTS = 100
# Stimulus singled out for the stim-#8 derivative figures (1-indexed: the 8th
# DMSO pulse overall — the 3rd pulse of the 2nd train).
STIM8_INDEX = 8


def _channel_peak_segments(state, exp_name, ch, cfg, grid, *,
                           responder_mask=None, stim_indices=None):
    """Resampled dF/F0 peak segments for one channel.

    Returns a list with one ``(n_cells, GRID_POINTS)`` array per stimulus —
    every cell's dF/F0 from the stim onset to ``SEGMENT_MINUTES`` later,
    resampled onto ``grid`` (relative minutes since onset). Grid points beyond
    a segment's actual extent (e.g. the final stim near the experiment end)
    are left NaN so they drop out of the mean.

    ``responder_mask`` (aligned to the corrected-lum ``CellID`` order) filters
    the segments to responder cells only when given.

    ``stim_indices`` (a set/sequence of 0-based stimulus indices) restricts the
    result to those stimuli — used by the stim-#8 derivative figures.
    """
    stim_frames = cfg["stim_frames"][ch]
    df = lum_dict_to_df(
        state["corrected_lum"][exp_name][ch]
    ).set_index("CellID")
    frame_cols = sorted(
        [c for c in df.columns if str(c).startswith("f")],
        key=lambda c: int(str(c).lstrip("f")),
    )
    if not frame_cols:
        return []
    frame_nums = np.array([int(str(c).lstrip("f")) for c in frame_cols])
    frame_to_col = {int(f): i for i, f in enumerate(frame_nums)}
    mat = df[frame_cols].values

    F0, _, _ = compute_f0_baseline(state, exp_name, ch, cfg)
    F0_safe = np.where(F0 == 0, np.nan, F0)
    dff = (mat - F0) / F0_safe
    if responder_mask is not None:
        dff = dff[np.asarray(responder_mask, dtype=bool)]
    minutes = frames_to_min(state, exp_name, ch, frame_nums)

    segments = []
    for si, sf in enumerate(stim_frames):
        if stim_indices is not None and si not in stim_indices:
            continue
        start_col = frame_to_col.get(int(sf))
        if start_col is None:
            continue
        stim_min = minutes[start_col]
        seg_cols = np.where(
            (minutes >= stim_min) & (minutes <= stim_min + SEGMENT_MINUTES)
        )[0]
        if seg_cols.size < 2:
            continue
        rel = minutes[seg_cols] - stim_min
        seg = dff[:, seg_cols]
        resampled = np.full((seg.shape[0], grid.size), np.nan)
        for i in range(seg.shape[0]):
            resampled[i] = np.interp(
                grid, rel, seg[i], left=np.nan, right=np.nan,
            )
        segments.append(resampled)
    return segments


def _pooled_stim8_segments(state, exp_name, cfg, grid, responder_masks):
    """Responder dF/F0 segments for stimulus ``STIM8_INDEX``, pooled per channel.

    Returns a stacked ``(n_segments, GRID_POINTS)`` array — one row per
    responder cell per channel for the single stim-#8 response window — or
    ``None`` when the stimulus is absent or has no responders.
    """
    rows = []
    for ch in cfg["channels"]:
        ch_mask = responder_masks.get((exp_name, ch))
        if ch_mask is None or not np.any(ch_mask):
            continue
        rows.extend(
            _channel_peak_segments(
                state, exp_name, ch, cfg, grid,
                responder_mask=ch_mask, stim_indices=(STIM8_INDEX - 1,),
            )
        )
    if not rows:
        return None
    return np.vstack(rows)


def analyze(experiments, state):
    """Cache RAW stacked average-peak segments per DMSO experiment.

    DMSO experiments are those with ``response_direction == "increase"``;
    everything else is a no-op (mirrors the source's skip). Per experiment the
    all-cells, responders-only, and stim-#8 responder segments are stacked and
    cached without the render-side coverage clip.
    """
    grid = np.linspace(0.0, SEGMENT_MINUTES, GRID_POINTS)
    responder_masks = get_responder_masks(experiments, state)

    for exp_name, cfg in experiments.items():
        if cfg.get("response_direction") != "increase":
            continue  # DMSO-only

        n_channels = len(cfg["channels"])
        all_segments = []
        resp_segments = []
        for ch in cfg["channels"]:
            all_segments.extend(
                _channel_peak_segments(state, exp_name, ch, cfg, grid)
            )
            ch_mask = responder_masks.get((exp_name, ch))
            if ch_mask is not None and np.any(ch_mask):
                resp_segments.extend(
                    _channel_peak_segments(
                        state, exp_name, ch, cfg, grid,
                        responder_mask=ch_mask,
                    )
                )

        if not all_segments:
            print(
                f"{exp_name}: no peak segments — "
                "skipping average-peak cache."
            )
            continue

        all_stacked = np.vstack(all_segments)
        resp_stacked = np.vstack(resp_segments) if resp_segments else None
        stim8_stacked = _pooled_stim8_segments(
            state, exp_name, cfg, grid, responder_masks
        )

        data = {
            "grid": grid,
            "all_segments_stacked": all_stacked,
            "resp_segments_stacked": resp_stacked,
            "stim8_stacked": stim8_stacked,
            "n_channels": int(n_channels),
        }
        meta = {
            "cell_line": cell_line_label(exp_name),
            "exp_name": exp_name,
            "n_seg_all": int(all_stacked.shape[0]),
            "n_seg_resp": (0 if resp_stacked is None
                           else int(resp_stacked.shape[0])),
            "n_seg_stim8": (0 if stim8_stacked is None
                            else int(stim8_stacked.shape[0])),
        }
        save_analysis_cache(data, exp_name, "average_peak", meta=meta)
        print(
            f"  cached average_peak.pkl for {exp_name} "
            f"({meta['n_seg_all']} all / {meta['n_seg_resp']} responder / "
            f"{meta['n_seg_stim8']} stim-#{STIM8_INDEX} segments)"
        )


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
