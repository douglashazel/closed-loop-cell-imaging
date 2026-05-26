"""End-to-end state preparation for every analysis script.

`prepare_state()` runs the four prep steps from the source script's `main()`:
    1. resolve_all_stim_frames
    2. compute_background_correction (cache hit on second run)
    3. build_frame_to_minutes_lookups
    4. clip_experiments_to_time_window

Each analysis script calls this once and consumes the returned state. The
per-experiment background pickle cache (April28_preprint_results/bg_cache/<exp>.pkl)
makes repeated invocations cheap.
"""

import os
import pickle
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from tqdm.auto import tqdm

from common.bg_fit import (
    build_background_sampler,
    eval_background_at_points,
    eval_background_image,
    fit_background_from_image,
)
from common.config import BG_FIT, CACHE_DIR, RECOMPUTE_BG
from common.io_paths import channel_dir, load_segmentation, sorted_image_files
from common.stim_resolve import resolve_all_stim_frames
from common.time_axis import (
    build_frame_to_minutes_lookups,
    clip_experiments_to_time_window,
)

sys.path.insert(0, "SCRIPTS")
from io_utils import load_msgpack  # noqa: E402


# Bump when the cache pickle layout or pipeline semantics change (e.g. when
# ``filter_dead_frames`` started gating in-cache BG-coef interpolation, or
# when ``bad_frames_file`` replaced auto-detection for an experiment).
PIPELINE_VERSION = 3


def detect_dead_frame_indices(bg_trace, *, mad_k=6.0, window=21):
    """Return frame indices where ``bg_trace`` diverges from rolling median.

    Uses a robust median-absolute-deviation test against a rolling median over
    ``window`` frames. Same threshold logic used inside ``mask_dead_frames``;
    factored out so the BG-fit pass can interpolate over the same indices
    before per-cell correction runs.
    """
    bg = np.asarray(bg_trace, dtype=np.float64)
    if bg.size < 3:
        return np.array([], dtype=np.int64)
    med = median_filter(bg, size=window, mode="nearest")
    delta = bg - med
    mad = float(np.median(np.abs(delta - np.median(delta))))
    scale = 1.4826 * mad if mad > 0 else 0.0
    if scale == 0.0:
        return np.array([], dtype=np.int64)
    return np.where(np.abs(delta) > mad_k * scale)[0]


def load_explicit_bad_frames(path):
    """Parse a bad-frames txt file into a sorted, deduped int array of indices.

    File format: a header line, then one line per bad frame whose first
    whitespace-separated token is the 0-based frame index. Any trailing
    columns (e.g. a ``light``/``dark`` label) are ignored — masking treats
    flashes and dropouts identically. Non-numeric lines (the header) are
    skipped.
    """
    idx = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            try:
                idx.append(int(tok[0]))
            except ValueError:
                continue  # header or other non-numeric line
    return np.array(sorted(set(idx)), dtype=np.int64)


def resolve_dead_frame_indices(cfg, bg_trace, *, mad_k=6.0, window=21):
    """Frame indices to treat as dead for ``cfg``'s experiment.

    When ``cfg["bad_frames_file"]`` is set, the explicit frame list in that
    file is used verbatim (clipped to the valid frame range) and no automatic
    MAD-based detection runs. Otherwise the indices are auto-detected from
    ``bg_trace`` via :func:`detect_dead_frame_indices`.
    """
    bad_file = cfg.get("bad_frames_file")
    if bad_file:
        idx = load_explicit_bad_frames(bad_file)
        n = len(np.asarray(bg_trace))
        return idx[(idx >= 0) & (idx < n)]
    return detect_dead_frame_indices(bg_trace, mad_k=mad_k, window=window)


def _interpolate_bg_at_dead_frames(coefs_by_frame, bg_min_by_frame, dead_idx):
    """Replace per-frame BG coefs/min at ``dead_idx`` with neighbor-interpolated values.

    Each frame's polynomial fit is computed against that frame's image. Dead
    frames (dropouts/flashes) produce anomalous polynomial coefficients that
    would feed bad bg_val estimates into the per-cell correction loop. After
    detecting the dead-frame indices from ``bg_trace``, interpolate every
    polynomial coefficient (and ``bg_min``) linearly across the surrounding
    non-dead frames so the per-cell pass uses a clean BG estimate at those
    indices. ``corrected_lum`` entries on dead frames are still NaN-masked
    later in ``mask_dead_frames`` (the cell's raw luminosity is still bad).
    """
    n_frames = coefs_by_frame.shape[0]
    if dead_idx.size == 0 or dead_idx.size >= n_frames:
        return
    good = np.ones(n_frames, dtype=bool)
    good[dead_idx] = False
    good_idx = np.where(good)[0]
    for k in range(coefs_by_frame.shape[1]):
        coefs_by_frame[dead_idx, k] = np.interp(
            dead_idx, good_idx, coefs_by_frame[good_idx, k]
        )
    bg_min_by_frame[dead_idx] = np.interp(
        dead_idx, good_idx, bg_min_by_frame[good_idx]
    )


def _bg_cache_path(exp_name, cfg):
    """Cache path keyed by whether dead-frame interpolation is enabled.

    Experiments with ``filter_dead_frames=False`` (or unset) keep the legacy
    unsuffixed path so existing caches stay valid. PC3 (and any future
    experiment that opts in) gets a separate ``__filterdead`` pickle so
    toggling the flag never silently reuses a stale cache.
    """
    suffix = "__filterdead" if cfg.get("filter_dead_frames") else ""
    return os.path.join(CACHE_DIR, f"{exp_name}{suffix}.pkl")


def compute_background_correction(experiments, recompute=RECOMPUTE_BG):
    """Run per-frame sampled polynomial background correction for every experiment.

    Caches per-experiment results to ``CACHE_DIR/<exp>[__filterdead].pkl`` and
    reuses them unless ``recompute`` is True or ``BG_FIT`` has changed since
    the cache was written. When ``cfg["filter_dead_frames"]`` is True, the
    per-frame polynomial coefficients are detected-and-interpolated across
    dead frames before the per-cell correction loop runs, so anomalous frame
    fits don't leak into ``corrected_lum`` or the cached BG maps.
    """
    state = {
        "corrected_lum": {},
        "bg_trace": {},
        "traj_by_channel": {},
        "frame_counts": {},
        "probe_data": {},
        "bg_map_by_ch": {},
        "bg_sampler_by_exp": {},
        "bg_coefs_by_ch": {},
        "bg_min_by_ch": {},
        "bg_sample_points_exp": {},
    }

    for exp_name, cfg in experiments.items():
        cache_path = _bg_cache_path(exp_name, cfg)

        if not recompute and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            if blob.get("BG_FIT") != BG_FIT:
                print(
                    f"=== {exp_name} === cached BG_FIT differs from current — recomputing."
                )
            elif blob.get("PIPELINE_VERSION") != PIPELINE_VERSION:
                print(
                    f"=== {exp_name} === cached PIPELINE_VERSION="
                    f"{blob.get('PIPELINE_VERSION')} differs from "
                    f"{PIPELINE_VERSION} — recomputing."
                )
            else:
                state["corrected_lum"][exp_name] = blob["corrected_lum"]
                state["bg_trace"][exp_name] = blob["bg_trace"]
                state["traj_by_channel"][exp_name] = blob["traj_by_channel"]
                state["frame_counts"][exp_name] = blob["frame_counts"]
                state["probe_data"][exp_name] = blob["probe_data"]
                state["bg_map_by_ch"][exp_name] = blob["bg_map_by_ch"]
                state["bg_coefs_by_ch"][exp_name] = blob["bg_coefs_by_ch"]
                state["bg_min_by_ch"][exp_name] = blob["bg_min_by_ch"]
                state["bg_sampler_by_exp"][exp_name] = blob["sampler"]
                state["bg_sample_points_exp"][exp_name] = blob["sampler"]["sample_points"]
                print(
                    f"=== {exp_name} === loaded cache from {cache_path} "
                    f"({len(blob['corrected_lum'])} channels)"
                )
                continue

        # No usable cache — run the full computation.
        state["corrected_lum"][exp_name] = {}
        state["bg_trace"][exp_name] = {}
        state["traj_by_channel"][exp_name] = {}
        state["frame_counts"][exp_name] = {}
        state["probe_data"][exp_name] = {}
        state["bg_map_by_ch"][exp_name] = {}
        state["bg_coefs_by_ch"][exp_name] = {}
        state["bg_min_by_ch"][exp_name] = {}

        sampler = build_background_sampler(cfg)
        state["bg_sampler_by_exp"][exp_name] = sampler
        state["bg_sample_points_exp"][exp_name] = sampler["sample_points"]
        H, W = sampler["shape"]
        print()
        print(
            f"=== {exp_name} === {len(sampler['sample_points'])} / "
            f"{BG_FIT['grid_n']**2} grid points snapped to cell-free background"
        )

        for ch in cfg["channels"]:
            fdir = os.path.join(channel_dir(cfg, ch), "frames")
            adir = os.path.join(channel_dir(cfg, ch), "analysis")
            ffiles = sorted_image_files(fdir)
            n_frames = len(ffiles)

            coefs_by_frame = np.zeros(
                (n_frames, sampler["A_pinv"].shape[0]), dtype=np.float64
            )
            bg_min_by_frame = np.zeros(n_frames, dtype=np.float64)
            sampled_bg_mean = np.zeros(n_frames, dtype=np.float32)
            sampled_bg_range = np.zeros((n_frames, 2), dtype=np.float32)

            probe_idx = n_frames // 2
            probe = None
            probe_bg = None
            probe_means = None

            for i, fname in enumerate(
                tqdm(ffiles, desc=f"{exp_name} / {ch} sampled bg", leave=False)
            ):
                img = np.array(
                    Image.open(os.path.join(fdir, fname)).convert("L"),
                    dtype=np.float32,
                )
                if img.shape != (H, W):
                    raise ValueError(
                        f"{exp_name} / {ch} frame {fname} shape {img.shape} "
                        f"does not match sampler shape {(H, W)}"
                    )

                coefs, bg_min, means = fit_background_from_image(img, sampler)
                coefs_by_frame[i] = coefs
                bg_min_by_frame[i] = bg_min
                sampled_bg_mean[i] = float(means.mean())
                sampled_bg_range[i] = (float(means.min()), float(means.max()))

                if i == probe_idx:
                    probe = img
                    probe_means = means
                    probe_bg = eval_background_image(
                        coefs, bg_min, (H, W), sampler["degree"]
                    )

            if cfg.get("filter_dead_frames"):
                dead_idx = resolve_dead_frame_indices(cfg, sampled_bg_mean)
                if dead_idx.size:
                    _interpolate_bg_at_dead_frames(
                        coefs_by_frame, bg_min_by_frame, dead_idx
                    )
                    print(
                        f"  {ch}: interpolated BG coefs at "
                        f"{dead_idx.size} dead frames during fit pass"
                    )
                    if probe_idx in dead_idx:
                        probe_bg = eval_background_image(
                            coefs_by_frame[probe_idx],
                            bg_min_by_frame[probe_idx],
                            (H, W),
                            sampler["degree"],
                        )

            lum = load_msgpack(os.path.join(adir, "luminosity_complete.json"))
            traj = load_msgpack(os.path.join(adir, "trajectories_complete.json"))

            corr_ch = {}
            for cid, frames in lum.items():
                coords = traj.get(cid, {})
                c_corr = {}
                for fk, v in frames.items():
                    if v is None or not fk.startswith("f"):
                        continue
                    idx = int(fk[1:])
                    if idx >= n_frames:
                        continue
                    x = coords.get(f"x{idx}")
                    y = coords.get(f"y{idx}")
                    if x is None or y is None:
                        continue
                    try:
                        xi = float(x)
                        yi = float(y)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= yi < H and 0 <= xi < W:
                        bg_val = eval_background_at_points(
                            coefs_by_frame[idx],
                            bg_min_by_frame[idx],
                            [xi],
                            [yi],
                            (H, W),
                            sampler["degree"],
                        )[0]
                        c_corr[fk] = float(v) - float(bg_val)
                if c_corr:
                    corr_ch[cid] = c_corr

            state["corrected_lum"][exp_name][ch] = corr_ch
            state["bg_trace"][exp_name][ch] = sampled_bg_mean
            state["traj_by_channel"][exp_name][ch] = traj
            state["frame_counts"][exp_name][ch] = n_frames
            state["bg_map_by_ch"][exp_name][ch] = probe_bg
            state["bg_coefs_by_ch"][exp_name][ch] = coefs_by_frame
            state["bg_min_by_ch"][exp_name][ch] = bg_min_by_frame

            state["probe_data"][exp_name][ch] = {
                "probe": probe,
                "probe_idx": probe_idx,
                "probe_bg": probe_bg,
                "corrected": probe - probe_bg,
                "probe_sample_mean": float(sampled_bg_mean[probe_idx]),
                "probe_sample_min": float(sampled_bg_range[probe_idx, 0]),
                "probe_sample_max": float(sampled_bg_range[probe_idx, 1]),
                "probe_bg_min": float(np.nanmin(probe_bg)),
                "probe_bg_max": float(np.nanmax(probe_bg)),
                "probe_means": probe_means,
            }
            print(
                f"  {ch}: {n_frames} frames | sample mean range "
                f"[{sampled_bg_mean.min():.1f}, {sampled_bg_mean.max():.1f}] | "
                f"probe bg max={np.nanmax(probe_bg):.2f} | "
                f"{len(corr_ch)} cells corrected"
            )

        blob = {
            "PIPELINE_VERSION": PIPELINE_VERSION,
            "BG_FIT": BG_FIT.copy(),
            "filter_dead_frames": bool(cfg.get("filter_dead_frames")),
            "sampler": sampler,
            "corrected_lum": state["corrected_lum"][exp_name],
            "bg_trace": state["bg_trace"][exp_name],
            "traj_by_channel": state["traj_by_channel"][exp_name],
            "frame_counts": state["frame_counts"][exp_name],
            "probe_data": state["probe_data"][exp_name],
            "bg_map_by_ch": state["bg_map_by_ch"][exp_name],
            "bg_coefs_by_ch": state["bg_coefs_by_ch"][exp_name],
            "bg_min_by_ch": state["bg_min_by_ch"][exp_name],
        }
        with open(cache_path, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  → cached to {cache_path}")

    print()
    print(
        "In-memory state ready: corrected_lum[exp][ch], bg_trace[exp][ch], "
        "traj_by_channel[exp][ch], frame_counts[exp][ch]."
    )
    return state


def mask_dead_frames(experiments, state, *, mad_k=6.0, window=21):
    """Mask frames where the sampled background diverges sharply from local median.

    Some cameras occasionally emit dark/dropped frames or flash frames that
    show up as sharp spikes in the per-cell traces. Dropouts depress the
    sampled background; flashes elevate it (and via bg subtraction, depress
    the corrected luminosity). The frame indices to mask come from
    :func:`resolve_dead_frame_indices` — an explicit ``bad_frames_file`` list
    when configured, otherwise a robust rolling-MAD test on the per-frame
    ``bg_trace``. Masked frames are set to NaN in
    ``state["corrected_lum"][exp][ch][cell]`` so that ``fill_dead_frames()``
    can reconstruct them via linear interpolation in the next pipeline step.
    Frame keys are preserved so column indices in any reconstructed
    cell-by-frame matrix continue to match actual frame indices.

    Opt-in per experiment via ``cfg["filter_dead_frames"] = True``.
    """
    for exp_name, cfg in experiments.items():
        if not cfg.get("filter_dead_frames"):
            continue
        for ch in cfg["channels"]:
            bg = np.asarray(
                state["bg_trace"][exp_name][ch], dtype=np.float64
            )
            dead_idx = resolve_dead_frame_indices(
                cfg, bg, mad_k=mad_k, window=window
            )
            if dead_idx.size == 0:
                print(
                    f"  {exp_name} / {ch}: no dead frames detected"
                )
                continue
            delta = bg - median_filter(bg, size=window, mode="nearest")
            n_neg = int(np.sum(delta[dead_idx] < 0))
            n_pos = int(np.sum(delta[dead_idx] > 0))
            dead_set = {f"f{int(i)}" for i in dead_idx}
            cells_touched = 0
            entries_masked = 0
            nan_val = float("nan")
            for cid, frames in state["corrected_lum"][exp_name][ch].items():
                hits = dead_set & frames.keys()
                if hits:
                    cells_touched += 1
                    entries_masked += len(hits)
                    for fk in hits:
                        frames[fk] = nan_val
            preview = ", ".join(str(int(i)) for i in dead_idx[:8])
            more = "" if dead_idx.size <= 8 else f", … (+{dead_idx.size - 8})"
            print(
                f"  {exp_name} / {ch}: masked {dead_idx.size} dead frames "
                f"({n_neg} dropouts + {n_pos} flashes) "
                f"[{preview}{more}] — masked {entries_masked} entries "
                f"across {cells_touched} cells"
            )


def fill_dead_frames(experiments, state):
    """Linearly interpolate masked (NaN) frames so traces are continuous.

    Runs after ``mask_dead_frames()``. For each cell, fills NaN values via
    ``np.interp`` between the nearest good frame indices; NaNs outside the
    range of good frames get the boundary value (np.interp's default).
    Produces a continuous trace for downstream plotting and analysis with
    no NaN-handling needed at the call sites.

    Opt-in per experiment via ``cfg["filter_dead_frames"] = True`` (same
    flag as ``mask_dead_frames``).
    """
    for exp_name, cfg in experiments.items():
        if not cfg.get("filter_dead_frames"):
            continue
        for ch in cfg["channels"]:
            cells_filled = 0
            entries_filled = 0
            for cid, frames in state["corrected_lum"][exp_name][ch].items():
                pairs = sorted(
                    ((int(k[1:]), v) for k, v in frames.items()),
                    key=lambda t: t[0],
                )
                idxs = np.array([p[0] for p in pairs], dtype=np.int64)
                vals = np.array([p[1] for p in pairs], dtype=np.float64)
                bad = np.isnan(vals)
                if not bad.any() or bad.all():
                    continue
                good_idxs = idxs[~bad]
                good_vals = vals[~bad]
                filled = np.interp(idxs[bad], good_idxs, good_vals)
                for k_idx, fv in zip(idxs[bad], filled):
                    frames[f"f{int(k_idx)}"] = float(fv)
                cells_filled += 1
                entries_filled += int(bad.sum())
            if entries_filled:
                print(
                    f"  {exp_name} / {ch}: linearly interpolated "
                    f"{entries_filled} masked entries across "
                    f"{cells_filled} cells"
                )


def apply_cell_mask_filter(experiments, state):
    """Drop cells whose frame-0 position lies on background of a per-channel filter mask.

    Opt-in per experiment via ``cfg["cell_mask_filter"]``, a mapping
    ``{channel: mask_path}`` where each path is relative to ``cfg["dir"]`` and
    points to a 2-D labeled mask. A cell is kept iff its trajectory has both
    ``x0`` and ``y0`` and the mask value at ``(round(y0), round(x0))`` is
    non-zero. Cells without frame-0 coordinates, with out-of-bounds positions,
    or that land on background are removed from both
    ``state["corrected_lum"][exp][ch]`` and ``state["traj_by_channel"][exp][ch]``,
    so every downstream analysis sees only the kept subset.

    Runs after ``compute_background_correction`` so the per-experiment BG
    cache stays valid — toggling the filter never invalidates the cache.
    """
    for exp_name, cfg in experiments.items():
        filt_by_ch = cfg.get("cell_mask_filter")
        if not filt_by_ch:
            continue
        for ch in cfg["channels"]:
            rel = filt_by_ch.get(ch)
            if rel is None:
                continue
            mask = load_segmentation(os.path.join(cfg["dir"], rel))
            H, W = mask.shape
            traj = state["traj_by_channel"][exp_name][ch]
            lum = state["corrected_lum"][exp_name][ch]

            keep = set()
            for cid, coords in traj.items():
                x = coords.get("x0")
                y = coords.get("y0")
                if x is None or y is None:
                    continue
                try:
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                except (TypeError, ValueError):
                    continue
                if not (0 <= xi < W and 0 <= yi < H):
                    continue
                if int(mask[yi, xi]) == 0:
                    continue
                keep.add(cid)

            before = len(lum)
            state["corrected_lum"][exp_name][ch] = {
                cid: v for cid, v in lum.items() if cid in keep
            }
            state["traj_by_channel"][exp_name][ch] = {
                cid: v for cid, v in traj.items() if cid in keep
            }
            print(
                f"  {exp_name} / {ch}: cell-mask filter kept "
                f"{len(state['corrected_lum'][exp_name][ch])} / {before} cells"
            )


def prepare_state(experiments, *, recompute_bg=False, check_direction=True):
    """Run the prep steps and return the populated state dict.

    Mirrors the prep block in april28_final_figures.py:main() — same
    operations, same order, identical resulting state — plus an opt-in
    dead-frame filter for experiments that set ``filter_dead_frames``.

    When ``check_direction`` is True (default), runs an empirical
    response-direction sanity check after dead-frame fill and logs a warning
    per (exp, ch) if the empirical sign disagrees with
    ``cfg["response_direction"]``. Diagnostic only — config still wins.
    """
    resolve_all_stim_frames(experiments)
    state = compute_background_correction(experiments, recompute=recompute_bg)
    apply_cell_mask_filter(experiments, state)
    build_frame_to_minutes_lookups(experiments, state)
    clip_experiments_to_time_window(experiments, state)
    mask_dead_frames(experiments, state)
    fill_dead_frames(experiments, state)
    if check_direction:
        from common.direction_check import confirm_response_direction
        state["direction_check"] = confirm_response_direction(experiments, state)
    return state
