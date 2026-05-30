#!/usr/bin/env python3
"""NRK hardware-feedback luminosity log analysis (no plotting).

NRK-only: for the NRK acid-feedback experiment, parses each channel's
``luminosity_log_channelN.json`` and computes the figure-ready intermediates the
``<ch>_hw_lum_log`` figures display:
  * per channel: frames→minutes axis, mean-luminosity trace, deduped acidic-pulse
    onset minutes, setpoint regions (start/end frames → minutes), the real-setpoint
    marker minute, and the x/y axis limits the figure uses.

For non-NRK selections this script silently no-ops (matches the original
``nrk_hardware_log.py`` behavior). The matplotlib rendering lives in
``plots/nrk_hardware_log.py``; this script writes only numbers and imports NO
matplotlib / style.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.cli import parse_args
from common.config import PULSE_DEDUP_FRAMES, cell_line_label
from common.io_paths import save_analysis_cache
from common.pipeline import prepare_state
from common.stim_resolve import _dedup_close_frames
from common.time_axis import frames_to_min, setpoint_regions_from_log

NAME = "nrk_hardware_log"
NRK_EXP = "nrk_acid_13APR26"


def _channel_hw_log(state, exp_name, ch, cfg):
    """Return the per-channel payload the hw-log figure needs, or ``None`` to skip.

    Mirrors the per-channel block of the original ``plot_nrk_hardware_log``: parse
    the luminosity JSON, build setpoint regions, dedup acidic pulses, convert all
    frame quantities to minutes, and pre-compute the x/y axis limits so the render
    is a pure transform.
    """
    log_path, ch_num = cfg["stim_logs"][ch]
    lum_log_path = os.path.join(
        os.path.dirname(log_path),
        f"luminosity_log_channel{ch_num}.json",
    )

    with open(lum_log_path) as f:
        entries = json.load(f)
    entries = [e for e in entries if e.get("channel") == ch_num]
    if not entries:
        print(f"NRK / {ch}: no entries in {lum_log_path} — skipping.")
        return None
    entries.sort(key=lambda e: e["frame"])

    regions = setpoint_regions_from_log(entries)
    if not regions:
        print(f"NRK / {ch}: no setpoint regions parsed — skipping.")
        return None

    frames = [e["frame"] for e in entries]
    luminosity = [e["mean_luminosity"] for e in entries]

    acid_frames_raw = [
        e["frame"] for e in entries if e.get("decision") == "add acidic media"
    ]
    acid_frames = _dedup_close_frames(acid_frames_raw, PULSE_DEDUP_FRAMES)

    frames_min = frames_to_min(state, exp_name, ch, frames)
    acid_min = (
        frames_to_min(state, exp_name, ch, acid_frames) if acid_frames else []
    )
    rsp = state["real_setpoint_min"][exp_name].get(ch)

    rsp_str = f"{rsp:.2f} min" if rsp is not None else "not detected"
    print(
        f"NRK / {ch}: {len(frames)} frames | "
        f"{len(acid_frames)} acidic pulses (raw {len(acid_frames_raw)}) | "
        f"{len(regions)} setpoint regions | real setpoint @ {rsp_str}"
    )

    # Setpoint regions: skip the initial calibration region (idx 0), keep the
    # palette index so the render colours each band identically. Convert the
    # start/end frames to minutes here (analysis owns the transform).
    setpoint_regions_min = []
    for idx, (start_f, end_f, sp) in enumerate(regions):
        if idx == 0:
            continue  # skip the initial calibration region
        start_m, end_m = frames_to_min(state, exp_name, ch, [start_f, end_f])
        setpoint_regions_min.append(
            {
                "idx": int(idx),
                "start_min": float(start_m),
                "end_min": float(end_m),
                "setpoint": float(sp),
            }
        )

    pulse_duration = float(cfg.get("stim_duration_minutes", 0.5) or 0.5)

    # Axis limits — verbatim from the original (in-window y range when available).
    x_lo = float(np.asarray(frames_min).min())
    x_hi = float(cfg.get("time_window_minutes", 30.0))
    in_window = [
        lum for lum, m in zip(luminosity, frames_min) if x_lo <= m <= x_hi
    ]
    if in_window:
        y_lo = min(in_window) - 2
        y_hi = max(in_window) * 1.05
    else:
        y_lo = min(luminosity) - 2
        y_hi = max(luminosity) * 1.05

    chamber = ch.split()[-1]

    return {
        "frames_min": np.asarray(frames_min, dtype=float),
        "luminosity": np.asarray(luminosity, dtype=float),
        "acid_min": np.asarray(acid_min, dtype=float),
        "pulse_duration": float(pulse_duration),
        "setpoint_regions_min": setpoint_regions_min,
        "real_setpoint_min": (None if rsp is None else float(rsp)),
        "chamber": chamber,
        "x_lim": (float(x_lo), float(x_hi)),
        "y_lim": (float(y_lo), float(y_hi)),
    }


def analyze(experiments, state, exp_name=NRK_EXP):
    """NRK-only: cache the per-channel hardware-log payloads (no-op otherwise)."""
    if exp_name not in experiments:
        return
    cfg = experiments[exp_name]

    per_channel = {}
    for ch in cfg["channels"]:
        d = _channel_hw_log(state, exp_name, ch, cfg)
        if d is not None:
            per_channel[ch] = d

    data = {"per_channel": per_channel}
    meta = {"cell_line": cell_line_label(exp_name), "exp_name": exp_name}
    save_analysis_cache(data, exp_name, NAME, meta=meta)
    print(f"  cached {NAME}.pkl for {exp_name} ({len(per_channel)} channels)")


def main():
    experiments, recompute_bg = parse_args()
    state = prepare_state(experiments, recompute_bg=recompute_bg)
    analyze(experiments, state)


if __name__ == "__main__":
    main()
