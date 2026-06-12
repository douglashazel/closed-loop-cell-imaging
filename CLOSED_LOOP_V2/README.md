# CLOSED_LOOP_V2 — live closed-loop microscopy / perturbation pipeline (web GUI)

A **browser-based** control system that runs a **closed feedback loop** on live
cells: it watches incoming microscope frames, segments cells with Cellpose,
measures per-cell mean fluorescence, compares each channel to a setpoint, and
drives an **ONIX hardware controller** (over HTTP) to dose acidic / neutral media
in response.

This is the same live acquisition + perturbation system as the sibling
`CLOSED_LOOP/` folder, but the napari desktop GUI (`LaunchNapari.py`) is replaced
by a **Flask web GUI** (`LaunchWebGUI.py`, served on port 5000) so the hub can be
driven from a browser — typically over an SSH port-forward. The
decision/actuation backend (`config.py`, `HandleSegmentations.py`,
`CreateDecisions.py`, `SendDecisions.py`, `MonitorPerformance.py`, `io_utils.py`,
`run_system.sh`) is otherwise the same. Like `CLOSED_LOOP/`, this is a
**separate, hardware-dependent system, not part of the reproducible figure
analysis** in `SCRIPTS/`.

> ⚠️ This pipeline requires lab hardware (an ONIX2 perfusion/microscopy server
> reachable over the network) and a CUDA GPU. It will not run end-to-end on a
> machine without them. The code is published for transparency/reuse; the figures
> in the root README are fully reproducible **without** it.

## Launch

```bash
cd CLOSED_LOOP_V2
python LaunchWebGUI.py            # serves http://0.0.0.0:5000
```

Then reach it from your local machine over an SSH tunnel:

```bash
ssh -L 5000:localhost:5000 user@ssh-host
# now open http://localhost:5000 in a browser
```

On first launch the server bootstraps `config.json` (from `config.py` defaults).
The browser UI then lets you edit the run config, preview frames/masks, run
Cellpose + push reference masks, set per-channel setpoints, tail the log, and
**Start/Stop** the pipeline (Start spawns `run_system.sh` as a process group).

## Files

| File | Role | Invocation |
|------|------|------------|
| `LaunchWebGUI.py` | **Entry point.** Flask "Closed-Loop Bio-Control Hub" web GUI on `0.0.0.0:5000`. Bootstraps `config.json`, serves the dashboard, and exposes the `/api/*` endpoints (config, setpoints, frames/masks, segmentation, luminosity plot, log tail, pipeline start/stop). Start/Stop spawns/kills `run_system.sh`. | `python LaunchWebGUI.py` |
| `templates/index.html` | Single-page dashboard served at `/`. | (served by `LaunchWebGUI.py`) |
| `static/app.js`, `static/style.css` | Browser client — polls the `/api/*` endpoints, renders frames/masks/plots and the readiness strip — plus styles. | (served by `LaunchWebGUI.py`) |
| `run_system.sh` | Orchestrator. Runs `config.py` to (re)build `config.json` + data dirs, then launches the monitor daemons in parallel and kills them all on exit/Ctrl-C. | `bash run_system.sh` (or via the GUI's Start button) |
| `config.py` | Config factory. Derives every data path from a single `global_path`; `build_config()` + `save_config()` write `config.json` and create the directories. Holds all run knobs. | `python config.py` (regenerates `config.json`) |
| `config.json` | Resolved run config written next to the scripts on launch. **Git-ignored** (embeds `global_path` + the ONIX endpoint). | (generated) |
| `HandleSegmentations.py` | Cellpose (GPU) segmentation daemon. Polls `watch_dir`, writes `{frame:05d}_channel{ch}.npy` masks + `_meta.json` ROI counts. Launched **only** when `continuous_segmentation=True`. | (launched by `run_system.sh`) |
| `CreateDecisions.py` | Decision stage. Reads frames + masks, computes per-cell mean fluorescence, compares to per-channel setpoints, writes per-channel decisions and an atomic `final_decisions/actions.toml`. | (launched by `run_system.sh`) |
| `SendDecisions.py` | Actuation stage. Consumes `actions.toml`, manages per-channel pulse timers, maps channel state → ONIX experiment (`NN`/`AN`/`NA`/`AA`), and drives the ONIX server over HTTP. Writes `media_status.json` + a timestamped hardware-telemetry CSV. | (launched by `run_system.sh`) |
| `MonitorPerformance.py` | Watches ROI-count metadata for per-channel cell gains/losses (flags >`threshold_ratio` changes) and periodically zips/cleans old files from `directories_to_clean`. | (launched by `run_system.sh`) |
| `io_utils.py` | Shared helpers (`log`, `load_config`, `parse_filename`, `wait_for_file`). Imported by the scripts above. | — |
| `system_overview.html` | Standalone single-page architecture reference for the whole system. | open in any browser |

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
media_status.json + ONIX_Hardware_Log_<ts>.csv   (read by the web GUI / Stage 2)
```

All working directories (`watch_dir`, `processed_masks`, `current_masks`,
`temp_decisions`, `final_decisions`, `flags`, …) live under `global_path` and are
created automatically by `config.py`.

## Configuration

Everything is driven by `config.py` → `config.json`. Edit `config.py` (or the run
config in the web UI) before first use. Key knobs:

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
> and the default values in `config.json` / `templates/index.html` with
> placeholders or environment variables, and scrub identifying values.

## ⚠️ Security / hardware

- `SendDecisions.py` issues HTTP requests that **create and run experiments on
  networked lab hardware** (perfusion + microscopy). Point it only at hardware you
  control, on a trusted network.
- `LaunchWebGUI.py` binds **`0.0.0.0:5000` with no authentication** and can
  Start/Stop the hardware-driving pipeline. **Do not expose port 5000 on an
  untrusted network** — keep it reachable only via `localhost` / an SSH tunnel.
  Start/Stop manage `run_system.sh` as a POSIX process group (`os.setsid` /
  `killpg`).

## Runtime artifacts

`config.json` is generated next to the scripts on launch and is **git-ignored**
(it embeds the resolved `global_path` and the ONIX endpoint) — do not commit it.
Acquired frames, masks, logs, and telemetry CSVs are written under `global_path`
and are not redistributed.

## Notes

- **Run from this directory.** The daemons load `config.json` via a relative path;
  `run_system.sh` and the web server `cd`/resolve here automatically. `io_utils.py`
  is imported as a bare module — keep these files co-located and **do not** put
  this directory on the same `sys.path` as `SCRIPTS/core_pipeline/` or the sibling
  `CLOSED_LOOP/` (each has its *own* `io_utils.py`).
- **GPU required** for segmentation: `CUDA_VISIBLE_DEVICES=0` and
  `CellposeModel(gpu=True)`.
- **POSIX-only** process management (`os.setsid` / `killpg`) in the launcher.
- Default mode is `continuous_segmentation=False` (frame-0 masks supplied
  externally); set it `True` to segment every frame via `HandleSegmentations.py`.
