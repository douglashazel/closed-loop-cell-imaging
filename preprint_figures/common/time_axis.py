"""Per-frame minutes lookup, time-window clipping, NRK setpoint regions.

Copied verbatim from april28_final_figures.py.
"""

import os

import numpy as np
import pandas as pd

from common.stim_resolve import (
    minutes_from_frame_mtimes,
    parse_monitor_log_frame_times,
    parse_monitor_log_setpoint_events,
)


def build_frame_to_minutes_lookups(experiments, state):
    """Populate ``state['frame_minutes_src']`` and ``state['real_setpoint_min']``."""
    state["frame_minutes_src"] = {}
    state["real_setpoint_min"] = {}

    for exp_name, cfg in experiments.items():
        state["frame_minutes_src"][exp_name] = {}
        state["real_setpoint_min"][exp_name] = {}

        for ch in cfg["channels"]:
            n_frames = state["frame_counts"][exp_name][ch]

            # ---- NRK: monitor.log -----------------------------------------
            if "stim_logs" in cfg and ch in cfg["stim_logs"]:
                log_path, ch_num = cfg["stim_logs"][ch]
                entries = parse_monitor_log_frame_times(log_path, ch_num)
                if not entries:
                    raise RuntimeError(
                        f"No frame log entries for {exp_name}/{ch} "
                        f"(channel {ch_num}) in {log_path}"
                    )
                t0 = entries[0][1]
                known_frames = np.asarray([e[0] for e in entries], dtype=float)
                known_minutes = np.asarray(
                    [(e[1] - t0).total_seconds() / 60.0 for e in entries],
                    dtype=float,
                )
                state["frame_minutes_src"][exp_name][ch] = (
                    known_frames, known_minutes,
                )

                sp_events = parse_monitor_log_setpoint_events(log_path, ch_num)
                if len(sp_events) >= 2:
                    real_sp_ts = sp_events[1][0]
                    state["real_setpoint_min"][exp_name][ch] = (
                        (real_sp_ts - t0).total_seconds() / 60.0
                    )
                    print(
                        f"  {exp_name} / {ch}: real setpoint @ "
                        f"{state['real_setpoint_min'][exp_name][ch]:.2f} min "
                        f"({len(sp_events)} setpoint events in log)"
                    )
                else:
                    state["real_setpoint_min"][exp_name][ch] = None
                    print(
                        f"  {exp_name} / {ch}: only {len(sp_events)} setpoint "
                        f"event(s) in monitor.log — no real-setpoint marker"
                    )

            # ---- C2C12 / PC3-with-CSV: timestamps.csv ---------------------
            elif "timestamps" in cfg and ch in cfg["timestamps"]:
                ts_path = os.path.join(cfg["dir"], cfg["timestamps"][ch])
                df = pd.read_csv(
                    ts_path, header=None,
                    names=["filename", "datetime", "minutes"],
                )
                minutes = df["minutes"].astype(float).values
                known_frames = np.arange(len(minutes), dtype=float)
                state["frame_minutes_src"][exp_name][ch] = (
                    known_frames, minutes,
                )
                state["real_setpoint_min"][exp_name][ch] = None

            # ---- single_channel_root w/o CSV: derive minutes from mtimes --
            elif cfg.get("single_channel_root"):
                known_frames, minutes = minutes_from_frame_mtimes(cfg, ch)
                state["frame_minutes_src"][exp_name][ch] = (
                    known_frames, minutes,
                )
                state["real_setpoint_min"][exp_name][ch] = None
                print(
                    f"  {exp_name} / {ch}: minutes from mtimes — "
                    f"frame 0 → {minutes[0]:.2f} min, "
                    f"frame {len(minutes) - 1} → {minutes[-1]:.2f} min"
                )

            # ---- Fallback: identity (frame == "minutes") ------------------
            else:
                print(
                    f"  WARNING: {exp_name}/{ch} has neither stim_logs nor "
                    f"timestamps — using frame index as the minutes axis"
                )
                idx = np.arange(n_frames, dtype=float)
                state["frame_minutes_src"][exp_name][ch] = (idx, idx.copy())
                state["real_setpoint_min"][exp_name][ch] = None

    print()


def clip_experiments_to_time_window(experiments, state):
    """Trim per-(experiment, channel) data to ``cfg['time_window_minutes']``."""
    for exp_name, cfg in experiments.items():
        window = cfg.get("time_window_minutes")
        if window is None:
            continue
        for ch in cfg["channels"]:
            n_frames = state["frame_counts"][exp_name][ch]
            all_min = frames_to_min(
                state, exp_name, ch, np.arange(n_frames, dtype=float),
            )
            kept_idx = np.where(all_min <= float(window))[0]
            if kept_idx.size == 0:
                raise RuntimeError(
                    f"{exp_name} / {ch}: time_window_minutes={window} "
                    f"excludes all frames"
                )
            cutoff = int(kept_idx[-1])
            kept_keys = {f"f{i}" for i in range(cutoff + 1)}

            corr = state["corrected_lum"][exp_name][ch]
            new_corr = {}
            for cid, per_cell in corr.items():
                kept = {k: v for k, v in per_cell.items() if k in kept_keys}
                if kept:
                    new_corr[cid] = kept
            state["corrected_lum"][exp_name][ch] = new_corr

            bg = state["bg_trace"][exp_name][ch]
            state["bg_trace"][exp_name][ch] = bg[: cutoff + 1]
            state["frame_counts"][exp_name][ch] = cutoff + 1

            sf = cfg["stim_frames"].get(ch, []) if isinstance(cfg["stim_frames"], dict) else []
            kept_sf = [f for f in sf if f <= cutoff]
            if isinstance(cfg["stim_frames"], dict):
                cfg["stim_frames"][ch] = kept_sf

            print(
                f"  {exp_name} / {ch}: clipped to ≤ {window:.1f} min — "
                f"frames 0..{cutoff} kept ({cutoff + 1} of {n_frames}), "
                f"{len(kept_sf)} stims remain"
            )
    print()


def frames_to_min(state, exp_name, ch, frames):
    """Look up minutes-from-t0 for an iterable / array of frame indices."""
    known_frames, known_minutes = state["frame_minutes_src"][exp_name][ch]
    fi = np.asarray(frames, dtype=float)
    return np.interp(fi, known_frames, known_minutes)


def setpoint_regions_from_log(entries):
    """Group consecutive log entries by setpoint.

    Returns a list of ``(start_frame, end_frame, setpoint)`` tuples.
    """
    regions = []
    if not entries:
        return regions
    start = entries[0]["frame"]
    sp = entries[0]["setpoint"]
    last = start
    for e in entries[1:]:
        if e["setpoint"] != sp:
            regions.append((start, last, sp))
            start = e["frame"]
            sp = e["setpoint"]
        last = e["frame"]
    regions.append((start, last, sp))
    return regions
