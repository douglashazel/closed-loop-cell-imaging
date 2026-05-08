"""Shared configuration for the preprint_figures pipeline.

Every value here is copied verbatim from april28_final_figures.py so the new
modular pipeline produces byte-identical output to the source script.
"""

import os


# =============================================================================
# Output / cache locations
# =============================================================================
OUT_ROOT = "April28_preprint_results"
CACHE_DIR = os.path.join(OUT_ROOT, "bg_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# Pipeline toggles
# =============================================================================
# Set to True to force a rebuild of the per-experiment background cache.
# Otherwise, cached pickles in ``CACHE_DIR`` are reused. Cache is also
# invalidated automatically whenever ``BG_FIT`` differs from the cached values.
RECOMPUTE_BG = False

# Peak frame for a given stimulus is ``stim_frame + PEAK_OFFSET``.
PEAK_OFFSET = 2

# Acidic-pulse dedup window. Any acidic-media request within this many frames
# of the previously-kept pulse is treated as a duplicate (the previous pulse
# is still being delivered — pulses last ~30 seconds). Tune for your frame
# rate: at 1 Hz, 30 frames ≈ 30 s.
PULSE_DEDUP_FRAMES = 10

# Marker string in monitoring.log that identifies a stimulus event.
ACID_ACTION = "add acidic media"

# Root path for the acid-feedback PE pipeline outputs (used by the NRK
# experiment to locate monitoring.log + luminosity_log_channelN.json).
PE_PIPELINE = "/mnt/exDisk1/douglashazel/DHcode/PE_Pipeline/V5"


# =============================================================================
# Background-fit configuration
# =============================================================================
BG_FIT = {
    "clearance_r": 40,        # required cell-free radius around each sample point
    "sample_r": 30,           # radius used for local background sample means
    "grid_n": 25,             # NxN candidate sampling grid
    "search_r": 180,          # max drift from grid point when snapping to clear pixels
    "n_mask_snapshots": 15,   # masks per channel used to build the cell-free map
    "poly_degree": 4,         # degree of smooth 2D polynomial background fit
    "min_grid_n": 160,        # coarse grid used to find the fitted-field minimum
}


# =============================================================================
# Learning-score configuration
# =============================================================================
LEARNING_STIMS_PER_TRAIN = 5  # Each DMSO train is 5 pulses (3 trains/expt).


# =============================================================================
# Experiment definitions
# =============================================================================
# stim_frames accepts:
#   - list[int]              → applied to every channel
#   - dict[ch, list[int]]    → per-channel override
#
# stim_logs (optional, overrides stim_frames for that channel):
#   dict[ch, (monitoring.log path, log channel number 1|2)]
#   Frames where action == 'add acidic media' are treated as stimuli.
#
# stim_minutes (optional, used with 'timestamps'):
#   list[float] minutes from perfusion start. Each channel's stim frame is the
#   frame whose absolute datetime is nearest to ``perfusion_start + stim_min``.
#   ``perfusion_start`` defaults to the earliest frame-0 datetime across the
#   channels listed in 'timestamps' (override with 'perfusion_start').
EXPERIMENTS = {
    "c2c12_dmso_09APR26": {
        "dir": "EXPERIMENTS/other/c2c12_dmso_pulses_perfusion_09APR26",
        "channels": ["channel 1", "channel 2", "channel 3"],
        "bg_ref": "EXPERIMENTS/other/background images for subtraction method/20APR26/8.png",
        "stim_frames": [],   # auto-filled by stim_minutes/timestamps below
        # F0 is computed automatically from stim_frames (mean of frames 0..first_stim-1).
        # 15 min neutral, then 3 blocks of (5x: 2 min DMSO + 8 min neutral) separated
        # by 30 min neutral. The first 15 min is treated as a single neutral block
        # (the priming sub-step is invisible at the cell side).
        "stim_minutes": [
            15, 25, 35, 45, 55,
            95, 105, 115, 125, 135,
            175, 185, 195, 205, 215,
        ],
        "stim_duration_minutes": 2.0,
        "stim_label": "DMSO pulse (2 min)",
        # DMSO drives luminosity *up* — the response is the post-stim maximum.
        "response_direction": "increase",
        "response_window": (1, 8),  # frames after stim: search for extremum in [stim+1, stim+8)
        "timestamps": {
            "channel 1": "timestamps/C2C12 DMSO perfusion 09APR26 channel 1 timestamps.csv",
            "channel 2": "timestamps/C2C12 DMSO perfusion 09APR26 channel 2 timestamps.csv",
            "channel 3": "timestamps/C2C12 DMSO perfusion 09APR26 channel 3 timestamps.csv",
        },
        # Optional: 'perfusion_start': '09-Apr-2026 14:08:16'
        # Default = earliest frame-0 datetime across channels listed in 'timestamps'.
    },
    "pc3_dmso_23MAR26": {
        "dir": "EXPERIMENTS/other/PC3 DMSO pulses perfusion 23MAR26",
        # Single-channel dataset: data lives at <dir>/{frames,masks,analysis}
        # directly. The "channel 1" name is purely a label (preserved as a
        # state key and in figure filenames); ``single_channel_root`` short-
        # circuits the per-channel subdir in path resolution.
        "channels": ["channel 1"],
        "single_channel_root": True,
        "bg_ref": "EXPERIMENTS/other/background images for subtraction method/20APR26/8.png",
        "stim_frames": [],   # auto-filled by stim_minutes/timestamps below
        # Same DMSO schedule as c2c12_dmso_09APR26 except for the acclimation
        # window: 10 min normal medium (vs 15 min for C2C12), then 3 trains of
        # 5 pulses (2 min DMSO + 8 min normal each), with 30 min rest between
        # trains. Total runtime 220 min. Per-frame timestamps live in a flat-
        # root CSV next to frames/ (not in a timestamps/ subfolder like C2C12).
        "stim_minutes": [
            10, 20, 30, 40, 50,
            90, 100, 110, 120, 130,
            170, 180, 190, 200, 210,
        ],
        "stim_duration_minutes": 2.0,
        "stim_label": "DMSO pulse (2 min)",
        "response_direction": "increase",
        "response_window": (1, 8),
        "timestamps": {
            "channel 1": "PC3 DMSO pulses perfusion 23MAR26 timestamps.csv",
        },
    },
    "nrk_acid_13APR26": {
        "dir": "EXPERIMENTS/other/nrk_acid_feedback_experiment_13APR26",
        "channels": ["channel 1 A", "channel 1 C", "channel 2 B", "channel 2 D"],
        "bg_ref": "EXPERIMENTS/other/background images for subtraction method/20APR26/8.png",
        "stim_frames": [],   # auto-filled by stim_logs below
        # F0 is computed automatically from stim_frames (mean of frames 0..first_stim-1).
        # Acid pulse delivery window is ~30 s.
        "stim_duration_minutes": 0.5,
        "stim_label": "Acid pulse (30 s)",
        # Acid drives luminosity *down* — the response is the post-stim minimum.
        "response_direction": "decrease",
        "response_window": (1, 8),
        # Only the first 30 min of NRK data are analyzed; everything past
        # this is dropped from corrected_lum/bg_trace/stim_frames before any
        # plot runs (see clip_experiments_to_time_window).
        "time_window_minutes": 30.0,
        "stim_logs": {
            "channel 1 A": (f"{PE_PIPELINE}/resultsApril13_exp2_channel1A_channel2B/monitoring.log", 1),
            "channel 2 B": (f"{PE_PIPELINE}/resultsApril13_exp2_channel1A_channel2B/monitoring.log", 2),
            "channel 1 C": (f"{PE_PIPELINE}/resultsApril13_exp3_channel1C_channel2D/monitoring.log", 1),
            "channel 2 D": (f"{PE_PIPELINE}/resultsApril13_exp3_channel1C_channel2D/monitoring.log", 2),
        },
    },
}
