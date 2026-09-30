# Stage 1 — core per-experiment pipeline

Turns raw microscope frames into per-cell fluorescence traces and dF/F0 for a
single experiment. Driven by the `run_*.sh` scripts at the project root, which
read their parameters from the experiment's `pipeline_config.yaml` (see the
main README, Stage 1); run the Python scripts directly **from the project root**.

| Script | Role | Invocation |
|--------|------|------------|
| `segmentation.py` | Cellpose segmentation: `frames/*.png` → `masks/*.npy` (written atomically; unreadable masks from a killed run are regenerated) | `python3 SCRIPTS/core_pipeline/segmentation.py --image_dir … --mask_dir … [--flow_threshold --cellprob_threshold --niter --diameter --cpu]` |
| `trajectories.py` | Link per-frame cells into trajectories; extract per-cell fluorescence | `python3 SCRIPTS/core_pipeline/trajectories.py --mask_dir … --image_dir … --save_path … [tracking params] [--workers --mask_timeout_sec --force_resume]` |
| `PreAnalysis.py` | QC plot of mean luminosity over time | `python3 SCRIPTS/core_pipeline/PreAnalysis.py --exp … --analysis_dir …` |
| `PostAnalysis.py` | QC only: spline background correction, derivative/STD, dF/F0 (see below) | `python3 SCRIPTS/core_pipeline/PostAnalysis.py --exp … --image_dir … --analysis_dir … [--f0_frame --stim_frames --recompute_bg]` |
| `io_utils.py` | Shared msgpack ⇄ DataFrame helpers (imported, not run) | — |
| `pipeline_config.py` | Reads and validates `pipeline_config.yaml` for the drivers, writes `analysis/run_history/`; TUNE_GUI saves the file with it | `python3 SCRIPTS/core_pipeline/pipeline_config.py check <experiment>` |
| `CreateGifsJson.py` | Optional per-cell contour GIFs | `python3 SCRIPTS/core_pipeline/CreateGifsJson.py --experiment_dir … [--max_cells --max_frames --crop_size]` |

`--diameter 0` (the default) lets Cellpose estimate the cell size.
Segmentation runs on the GPU when CUDA is available; `--cpu` forces the CPU.
`trajectories.py` polls for masks while segmentation runs. It gives up with a
non-zero exit when no new mask appears for `--mask_timeout_sec` (default 1 h),
and uses `--workers` processes (default: half the CPUs) to find cell centres.

### Tracking

Each track is predicted at its last seen position (plus the `--shift_xy` stage
shift once it crosses `--shift_frame`). Detections within `--max_distance`
pixels are assigned nearest first, and each detection goes to at most one
track. A track with no detection this frame is kept for `--grace_period`
frames; a detection with no track starts a new one. `--radius` (0 = off)
restricts tracking to a circular region. Frames are ordered and keyed by the
`timepoint_NNNNN` token in their filename, and a frame without that token is an
error. `--shift_frame` and `PostAnalysis.py --f0_frame` / `--stim_frames` refer
to that token.

### Output contract (written under each experiment's `analysis/`)

`trajectories_complete.json`, `luminosity_complete.json`,
`trajectories_complete.csv`, `luminosity_complete.csv`,
`luminosity_corrected_complete.json`, `bg_values_cache.npy`, `run_history/`
(one YAML record of the parameters per driver run, written by
`pipeline_config.py`; runs before it wrote `config.txt`), `run_params.json`,
`cellpose_centers/`, and `plots/`. Stage 2
(`SCRIPTS/preprint_analysis/`) reads only `trajectories_complete.json` and
`luminosity_complete.json`; the rest are for inspection and for
`PostAnalysis.py`. Don't rename these files.

> **The `.json` files are msgpack, not JSON.** Read them with
> `io_utils.load_msgpack` (a text editor or `json.load` will fail). The names
> are kept for compatibility with Stage 2 and the supplement exporter.

### Re-running with different parameters

`trajectories.py` resumes from `analysis/trajectories.json` and records its
tracking parameters in `analysis/run_params.json`. It refuses to resume a run
made with different parameters unless you pass `--force_resume`. To start over
with new parameters, delete the experiment's `analysis/` directory first. Do
not do this for the eight chambers published in the paper: their `analysis/`
outputs are the inputs to the published results. `PostAnalysis.py` recomputes
its background cache when the frame count or the trajectory file changes, or
with `--recompute_bg`.

### Stage 1 background and F0 are not the paper's

`PostAnalysis.py` is a quick-look QC tool. It fits a 6×6 bicubic spline to
each frame's background and takes F0 from a single frame (`--f0_frame`). The
published figures instead use Stage 2's per-frame 2-D polynomial background fit
(`SCRIPTS/preprint_analysis/common/bg_fit.py`, parameters `BG_FIT` in
`common/config.py`) and take F0 as each cell's mean over the frames before the
first stimulus. Numbers from `PostAnalysis.py` will therefore differ from the
figures.

### Notes

- `PreAnalysis.py`/`PostAnalysis.py`/`trajectories.py` import `io_utils` as a
  bare module — keep these files co-located in this directory.
- `PostAnalysis.py` writes `analysis/post_analysis_complete.txt` when it
  finishes; the TUNE_GUI progress view polls for it.
