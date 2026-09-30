# Stage 2 — preprint analysis

Pools the Stage-1 outputs of many experiments into the published figures,
statistics, and multi-panel mosaics of the preprint
(<https://doi.org/10.64898/2026.08.13.744473>).

The two driver scripts — `run_aggregate_results.sh` and `run_aggregate_plots.sh` — live at the
**project root** (alongside the Stage-1 `run_*.sh`); this directory holds the
Python they invoke. Run them from the project root (they `cd` there
automatically regardless).

## Two-step flow: analyze → cache → plot

```
run_aggregate_results.sh ──▶ analyze_*.py ──▶ <OUT_ROOT>/analysis_cache/<exp>/<analysis>.pkl
run_aggregate_plots.sh   ──▶ make_figures.py ─▶ <OUT_ROOT>/<exp>/*.png  (+ /mosaics/*.png)
```

1. **`run_aggregate_results.sh`** runs each `analyze_<name>.py` per experiment and caches
   figure-ready intermediates. `analyze_responders.py` **must run first** (it
   produces the shared responder masks every other analysis reads); the script
   enforces this order. Background-correction state is cached per experiment so
   repeat runs are fast.
2. **`run_aggregate_plots.sh`** runs `make_figures.py`, which loads each cache and renders
   standalone PNGs plus the named mosaics. Optionally runs
   `aggregate_preprint_pdf.py` (set `AGGREGATE_PDF=true`).
3. **`make_mosaic_captions.py`** (optional) writes a `.txt` caption beside each
   saved mosaic.

Both run scripts have a `CONFIG` block at the top to restrict the
analyses / experiments / figures / mosaics.

## Layout

- `common/` — shared configuration and analysis library. `config.py` holds the
  `EXPERIMENTS` registry and `OUT_ROOT` (the output directory name); `io_paths.py`
  derives all cache/figure paths from `OUT_ROOT`.
- `data/` — small non-image inputs (frame timestamps, the C2C12 cell-selection
  masks, the PC-3 bad-frame list, scrubbed NRK controller logs), referenced from
  `config.py` via `DATA_DIR`. See [`data/README.md`](data/README.md).
- `plots/` — figure render modules (one per analysis) + `mosaics.py`.
- `figures_spec.py`, `style.py` — figure metadata and the locked figure style.

## Supplementary data bundle

Two scripts publish and consume the per-chamber data bundle at `supplement/`
(tracked in git; see the top-level `README.md`):

- `export_supplement.py` — **maintainers only.** Writes `supplement/` from the
  warm `results/bg_cache/` pickles: raw + background-corrected fluorescence,
  dF/F0, cell positions, background, time axis, frame-0 masks, and a
  `metadata.json` per chamber, plus `index.json` and `CHECKSUMS.sha256`. It is
  read-only with respect to every existing pipeline output. Requires the raw
  experiment tree.
- `load_supplement.py` — **third parties.** Rebuilds the Stage-2 `state` dict
  from the exported tables and monkeypatches `common.pipeline.prepare_state`, so
  the `analyze_*.py` scripts run unchanged without the ~100 GB of raw frames:

  ```bash
  python SCRIPTS/preprint_analysis/load_supplement.py \
      --analyses responders dff average_peak correlation_distance \
                 clustering response_violins learning_scores nrk_hardware_log
  ./run_aggregate_plots.sh
  ```

  All seven mosaics build from the bundle. Only `responder_diagnostic` (its
  frame-sharpness panel) reads the images directly; it is skipped with a
  notice. Both scripts share an `EXPORT_VERSION` constant; the loader refuses a
  bundle it does not recognise.

`supplement_README.md` is the bundle's own documentation — `export_supplement.py`
copies it to `supplement/README.md`, so **edit it here**, not in `supplement/`.
The same goes for `load_supplement.py`. After editing either one without
re-running the export, copy it into `supplement/` and update only its line in
`supplement/CHECKSUMS.sha256`.

## Notes

- Imports of `common`/`plots` resolve via `sys.path` inserts computed from each
  file's location, so this directory can be moved freely — but the package names
  `common` and `plots` must not change.
- The analyses import `io_utils` from Stage 1 via
  `sys.path.insert(0, "SCRIPTS/core_pipeline")` (relative to the project root).
- `common/io_paths.py` stamps an `ANALYSIS_VERSION` into each cache and refuses to
  load a mismatched pickle — bump it when a cache schema changes. Version 2
  (learning-score population p-values cached) needs caches written by version 1
  to be regenerated before plotting.
- Random draws (the responder pseudo-stimulus null, permutation nulls) are
  seeded per experiment, so an analysis gives the same numbers whether
  experiments run in one process (`load_supplement.py`) or one per process
  (`run_aggregate_results.sh`).
- The NRK acid-feedback controller logs are read from `data/nrk_acid_13APR26/`
  (`stim_logs` in `common/config.py`).
