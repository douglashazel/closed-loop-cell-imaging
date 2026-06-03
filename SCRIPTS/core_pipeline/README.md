# Stage 1 — core per-experiment pipeline

Turns raw microscope frames into per-cell fluorescence traces and dF/F0 for a
single experiment. Driven by the `run_*.sh` scripts at the project root; run
everything **from the project root**.

| Script | Role | Invocation |
|--------|------|------------|
| `segmentation.py` | Cellpose (GPU) segmentation: `frames/*.png` → `masks/*.npy` | `python3 SCRIPTS/core_pipeline/segmentation.py --image_dir … --mask_dir … [--flow_threshold --cellprob_threshold --niter --diameter]` |
| `trajectories.py` | Link per-frame cells into trajectories; extract per-cell fluorescence | `python3 SCRIPTS/core_pipeline/trajectories.py --mask_dir … --image_dir … --save_path … [tracking params]` |
| `PreAnalysis.py` | QC plot of mean luminosity over time | `python3 SCRIPTS/core_pipeline/PreAnalysis.py --exp … --analysis_dir …` |
| `PostAnalysis.py` | Spline background correction, derivative/STD, dF/F0 | `python3 SCRIPTS/core_pipeline/PostAnalysis.py --exp … --image_dir … --analysis_dir … [--f0_frame --stim_frames]` |
| `io_utils.py` | Shared msgpack ⇄ DataFrame helpers (imported, not run) | — |
| `CreateGifs.py`, `CreateGifsJson.py` | Optional per-cell contour GIFs (edit the `CHANGE HERE` block, then run) | `python3 SCRIPTS/core_pipeline/CreateGifsJson.py` |

### Output contract (written under each experiment's `analysis/`)

`trajectories_complete.json`, `luminosity_complete.json`,
`trajectories_complete.csv`, `luminosity_complete.csv`,
`luminosity_corrected_complete.json`, `bg_values_cache.npy`, `config.txt`,
`cellpose_centers/`, and `plots/`. These filenames are a fixed contract consumed
by `PostAnalysis.py` and by Stage 2 (`SCRIPTS/preprint_analysis/`), so don't
rename them.

### Notes

- Frame ordering depends on a `timepoint_NNNNN` token in the frame filenames.
- `PreAnalysis.py`/`PostAnalysis.py` import `io_utils` as a bare module — keep
  these files co-located in this directory.
