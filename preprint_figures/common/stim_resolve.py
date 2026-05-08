"""Stim-frame resolution. Copied verbatim from april28_final_figures.py."""

import os
import re

import numpy as np
import pandas as pd

from common.config import ACID_ACTION, PULSE_DEDUP_FRAMES
from common.io_paths import channel_dir, sorted_image_files


def _dedup_close_frames(frames, min_gap):
    """Drop frames within ``min_gap`` of the previously kept frame."""
    if not frames:
        return []
    sorted_frames = sorted(set(frames))
    result = [sorted_frames[0]]
    for f in sorted_frames[1:]:
        if f - result[-1] > min_gap:
            result.append(f)
    return result


def parse_stim_frames_from_log(log_path, channel_num, action=ACID_ACTION):
    """Return frame indices for ``action`` events on ``channel_num``."""
    pat = re.compile(
        rf"(\d{{5}})_channel{channel_num}\s*:\s*[0-9.+\-eE]+\s*->\s*{re.escape(action)}"
    )
    frames = set()
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                frames.add(int(m.group(1)))
    return _dedup_close_frames(frames, PULSE_DEDUP_FRAMES)


def parse_monitor_log_frame_times(log_path, channel_num):
    """Return ``[(frame_idx, datetime), ...]`` per frame entry for ``channel_num``."""
    pat = re.compile(
        rf"^\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\]\s+"
        rf"(\d{{5}})_channel{channel_num}\s*:"
    )
    entries = []
    with open(log_path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ts = pd.to_datetime(m.group(1), format="%Y-%m-%d %H:%M:%S")
                frame = int(m.group(2))
                entries.append((frame, ts))
    entries.sort(key=lambda t: t[0])
    return entries


def parse_monitor_log_setpoint_events(log_path, channel_num):
    """Return ``[(datetime, value), ...]`` for ``Setpoint channelN: VALUE`` lines."""
    pat = re.compile(
        rf"^\[(\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}})\]\s+"
        rf"Setpoint channel{channel_num}\s*:\s*([0-9.+\-eE]+)"
    )
    events = []
    with open(log_path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ts = pd.to_datetime(m.group(1), format="%Y-%m-%d %H:%M:%S")
                events.append((ts, float(m.group(2))))
    events.sort(key=lambda t: t[0])
    return events


def parse_stim_frames_from_timestamps(ts_path, stim_minutes, perfusion_start_dt):
    """Map a list of minutes-since-perfusion-start to frame indices."""
    df = pd.read_csv(
        ts_path, header=None, names=["filename", "datetime", "minutes"]
    )
    dts = pd.to_datetime(df["datetime"].str.strip(), format="%d-%b-%Y %H:%M:%S")
    frames = []
    for sm in stim_minutes:
        target = perfusion_start_dt + pd.Timedelta(minutes=float(sm))
        idx = int((dts - target).abs().idxmin())
        frames.append(idx)
    return frames


def minutes_from_frame_mtimes(cfg, ch):
    """Return ``(known_frames, minutes)`` derived from frame file mtimes."""
    fdir = os.path.join(channel_dir(cfg, ch), "frames")
    ffiles = sorted_image_files(fdir)
    mtimes = np.asarray(
        [os.path.getmtime(os.path.join(fdir, f)) for f in ffiles],
        dtype=float,
    )
    deltas = np.diff(mtimes)
    if (deltas < 0).any():
        n_neg = int((deltas < 0).sum())
        print(
            f"  WARNING: mtimes under {fdir} have {n_neg} non-monotonic steps "
            "— clamping to 0."
        )
        deltas = np.clip(deltas, 0, None)
    minutes = np.concatenate([[0.0], np.cumsum(deltas) / 60.0])
    known_frames = np.arange(len(minutes), dtype=float)
    return known_frames, minutes


def resolve_all_stim_frames(experiments):
    """Mutate ``experiments`` in place so each ``cfg['stim_frames']`` is a dict.

    Precedence (highest to lowest):
        1. ``stim_logs[ch]`` (parsed monitoring.log)
        2. ``stim_minutes`` resolved via ``timestamps``
        3. dict ``stim_frames[ch]``
        4. list ``stim_frames`` broadcast to all channels
    """
    for exp_name, cfg in experiments.items():
        base = cfg.get("stim_frames", [])
        if isinstance(base, dict):
            resolved = {ch: list(base.get(ch, [])) for ch in cfg["channels"]}
        else:
            resolved = {ch: list(base) for ch in cfg["channels"]}

        if "stim_minutes" in cfg and "timestamps" in cfg:
            ts_paths = {
                ch: os.path.join(cfg["dir"], rel)
                for ch, rel in cfg["timestamps"].items()
            }
            if "perfusion_start" in cfg:
                perfusion_start = pd.to_datetime(
                    cfg["perfusion_start"], format="%d-%b-%Y %H:%M:%S"
                )
            else:
                first_ts = []
                for p in ts_paths.values():
                    first = pd.read_csv(
                        p,
                        header=None,
                        nrows=1,
                        names=["filename", "datetime", "minutes"],
                    )
                    first_ts.append(
                        pd.to_datetime(
                            first["datetime"].iloc[0].strip(),
                            format="%d-%b-%Y %H:%M:%S",
                        )
                    )
                perfusion_start = min(first_ts)
            for ch, p in ts_paths.items():
                if ch not in cfg["channels"]:
                    print(f"  warning: timestamps references unknown channel '{ch}'")
                    continue
                frames = parse_stim_frames_from_timestamps(
                    p, cfg["stim_minutes"], perfusion_start
                )
                resolved[ch] = frames
                print(
                    f"  {exp_name} / {ch}: resolved {len(frames)} stims from "
                    f"timestamps (perfusion_start={perfusion_start}) → {frames}"
                )

        if (
            "stim_minutes" in cfg
            and cfg.get("single_channel_root")
            and "timestamps" not in cfg
        ):
            for ch in cfg["channels"]:
                _, minutes = minutes_from_frame_mtimes(cfg, ch)
                frames = [
                    int(np.argmin(np.abs(minutes - float(sm))))
                    for sm in cfg["stim_minutes"]
                ]
                resolved[ch] = frames
                print(
                    f"  {exp_name} / {ch}: resolved {len(frames)} stims from "
                    f"mtimes-derived minutes → {frames}"
                )

        for ch, (log_path, ch_num) in cfg.get("stim_logs", {}).items():
            if ch not in cfg["channels"]:
                print(f"  warning: stim_logs references unknown channel '{ch}'")
                continue
            frames = parse_stim_frames_from_log(log_path, ch_num)
            resolved[ch] = frames
            print(
                f"  {exp_name} / {ch}: parsed {len(frames)} stims from "
                f"{os.path.basename(os.path.dirname(log_path))}/monitoring.log "
                f"(log channel {ch_num})"
            )

        cfg["stim_frames"] = resolved

    print()
    for exp_name, cfg in experiments.items():
        for ch in cfg["channels"]:
            sf = cfg["stim_frames"][ch]
            preview = f"{sf[:3]}...{sf[-3:]}" if len(sf) > 6 else str(sf)
            print(
                f"  {exp_name:25s} / {ch:12s} → {len(sf):3d} stims  {preview}"
            )
