# Stage 2 — preprint analysis

Pools the Stage-1 outputs of many experiments into the published figures,
statistics, and multi-panel mosaics. Run **from the project root** (the shell
scripts `cd` there automatically).

## Two-step flow: analyze → cache → plot

```
run_analysis.sh ──▶ analyze_*.py ──▶ <OUT_ROOT>/analysis_cache/<exp>/<analysis>.pkl
run_plots.sh    ──▶ make_figures.py ─▶ <OUT_ROOT>/<exp>/*.png  (+ /mosaics/*.png)
```

1. **`run_analysis.sh`** runs each `analyze_<name>.py` per experiment and caches
   figure-ready intermediates. `analyze_responders.py` **must run first** (it
   produces the shared responder masks every other analysis reads); the script
   enforces this order. Background-correction state is cached per experiment so
   repeat runs are fast.
2. **`run_plots.sh`** runs `make_figures.py`, which loads each cache and renders
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
- `plots/` — figure render modules (one per analysis) + `mosaics.py`.
- `figures_spec.py`, `style.py` — figure metadata and the locked figure style.

## Notes

- Imports of `common`/`plots` resolve via `sys.path` inserts computed from each
  file's location, so this directory can be moved freely — but the package names
  `common` and `plots` must not change.
- The analyses import `io_utils` from Stage 1 via
  `sys.path.insert(0, "SCRIPTS/core_pipeline")` (relative to the project root).
- `common/io_paths.py` stamps an `ANALYSIS_VERSION` into each cache and refuses to
  load a mismatched pickle — bump it when a cache schema changes.
- For the NRK acid-feedback experiment, set `PE_PIPELINE` (env var) to the
  external feedback-pipeline output directory.
