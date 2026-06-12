# CLOSED_LOOP — live closed-loop microscopy / perturbation pipeline

A napari-based control system that runs a **closed feedback loop** on live cells:
it watches incoming microscope frames, segments cells with Cellpose, measures
per-cell mean fluorescence, compares each channel to a setpoint, and drives an
**ONIX hardware controller** (over HTTP) to dose acidic / neutral media in
response. This is the live acquisition + perturbation system that *generated* the
acid-feedback experiments analysed by the offline pipeline in `SCRIPTS/`; it is a
**separate, hardware-dependent system, not part of the reproducible figure
analysis**.

> ⚠️ This pipeline requires lab hardware (an ONIX2 perfusion/microscopy server
> reachable over the network) and a CUDA GPU. It will not run end-to-end on a
> machine without them. The code is published for transparency/reuse; the figures
> in the root README are fully reproducible **without** it.

## Files

| File | Role | Invocation |
|------|------|------------|
| `LaunchNapari.py` | **Entry point.** napari "Closed-Loop Bio-Control Hub" GUI: edit run config, preview frames/masks, run Cellpose + ROI filtering, tail logs, and Start/Stop the pipeline. Bootstraps `config.json` on startup, then spawns `run_system.sh`. | `python LaunchNapari.py` |
| `run_system.sh` | Orchestrator. Runs `config.py` to (re)build `config.json` + data dirs, then launches the monitor daemons in parallel and kills them all on exit/Ctrl-C. | `bash run_system.sh` (or via the GUI's Start button) |
| `config.py` | Config factory. Derives every data path from a single `global_path`; `build_config()` + `save_config()` write `config.json` and create the directories. Holds all run knobs. | `python config.py` (regenerates `config.json`) |
| `HandleSegmentations.py` | Cellpose (GPU) segmentation daemon. Polls `watch_dir`, writes `{frame:05d}_channel{ch}.npy` masks + `_meta.json` ROI counts. Launched **only** when `continuous_segmentation=True`. | (launched by `run_system.sh`) |
| `CreateDecisions.py` | Decision stage. Reads frames + masks, computes per-cell mean fluorescence, compares to per-channel setpoints, writes per-channel decisions and an atomic `final_decisions/actions.toml`. | (launched by `run_system.sh`) |
| `SendDecisions.py` | Actuation stage. Consumes `actions.toml`, manages per-channel pulse timers, maps channel state → ONIX experiment (`NN`/`AN`/`NA`/`AA`), and drives the ONIX server over HTTP. Writes `media_status.json` + a timestamped hardware-telemetry CSV. | (launched by `run_system.sh`) |
| `MonitorPerformance.py` | Watches ROI-count metadata for per-channel cell gains/losses (flags >`threshold_ratio` changes) and periodically zips/cleans old files from `directories_to_clean`. | (launched by `run_system.sh`) |
| `io_utils.py` | Shared helpers (`log`, `load_config`, `parse_filename`, `wait_for_file`). Imported by the scripts above. | — |
| `debug_onix.py` | Manual diagnostic for the ONIX HTTP server (status / open / close / create probes). Not part of the run path. | `python debug_onix.py [status\|open\|close\|create\|...]` |

## Data flow

```
microscope frames ─▶ watch_dir
        │
        ▼  (Cellpose masks: frame-0, or continuous)
CreateDecisions.py ─▶ per-cell fluorescence vs setpoint ─▶ final_decisions/actions.toml
        │
        ▼
SendDecisions.py ─▶ ONIX2 server (HTTP) ─▶ dose acidic / neutral media
        │
        ▼
media_status.json + ONIX_Hardware_Log_<ts>.csv   (read by the napari GUI / Stage 2)
```

All working directories (`watch_dir`, `processed_masks`, `current_masks`,
`temp_decisions`, `final_decisions`, `flags`, …) live under `global_path` and are
created automatically by `config.py`.

## Configuration

Everything is driven by `config.py` → `config.json`. Edit `config.py` (or the run
config in the GUI) before first use. Key knobs:

- `global_path` — root for all data directories (**machine-specific**).
- `watch_dir` — directory of incoming frames (defaults to one specific experiment).
- `num_channels`, `threshold_ratio`, `acidic_pulse_sec`, `run_duration_sec`,
  `continuous_segmentation`.
- `onix_server_ip` / `onix_server_port` — the ONIX2 hardware server endpoint.
- `experiment_templates` — Windows paths to the `.OnixExp` templates on the ONIX
  control PC.
- `decision_key` — media-action → integer mapping.

> **TODO (authors):** before public release, replace the machine-specific defaults
> in `config.py` (`global_path`, `onix_server_ip`/`port`, `experiment_templates`)
> and the `LOGFILE` path in `run_system.sh` with placeholders or environment
> variables, and scrub identifying values.

## ⚠️ Security / hardware

`SendDecisions.py` and `debug_onix.py` issue HTTP requests that **create and run
experiments on networked lab hardware** (perfusion + microscopy). Point them only
at hardware you control, on a trusted network. `LaunchNapari.py` also spawns
`run_system.sh` as a subprocess process group.

## Runtime artifacts

`config.json` is generated next to the scripts on launch and is **git-ignored**
(it embeds the resolved `global_path` and the ONIX endpoint) — do not commit it.
Acquired frames, masks, logs, and the `resultsApril*/` example-run directories are
written under `global_path` and are not redistributed.

## Notes

- **Run from this directory.** The daemons load `config.json` via a relative path;
  `run_system.sh` and the GUI `cd` here automatically. `io_utils.py` is imported as
  a bare module — keep these files co-located and **do not** put this directory and
  `SCRIPTS/core_pipeline/` on the same `sys.path` (that directory has a *different*
  `io_utils.py`).
- **GPU required** for segmentation: `CUDA_VISIBLE_DEVICES=0` and
  `CellposeModel(gpu=True)`.
- **POSIX-only** process management (`os.setsid` / `killpg`) in the GUI launcher.
- Default mode is `continuous_segmentation=False` (frame-0 masks supplied
  externally); set it `True` to segment every frame via `HandleSegmentations.py`.
