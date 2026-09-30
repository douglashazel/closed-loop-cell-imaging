# CLOSED_LOOP_GUI — live closed-loop microscopy / perturbation pipeline (web GUI)

A **browser-based** control system that runs a **closed feedback loop** on live
cells: it watches incoming microscope frames, segments cells with Cellpose,
measures per-cell mean fluorescence, compares each channel to a setpoint, and
drives an **ONIX hardware controller** (over HTTP) to dose acidic / neutral media
in response.

The hub is driven from a browser — typically over an SSH port-forward — by a
**Flask web GUI** (`LaunchWebGUI.py`, served on port 5000) sitting on top of a
daemon backend (`config.py`, `HandleSegmentations.py`, `CreateDecisions.py`,
`SendDecisions.py`, `MonitorPerformance.py`, `io_utils.py`, `run_system.sh`).
This is a **separate, hardware-dependent system, not part of the reproducible
figure analysis** in `SCRIPTS/`.

> ⚠️ This pipeline requires lab hardware (an ONIX2 perfusion/microscopy server
> reachable over the network) and a CUDA GPU. It will not run end-to-end on a
> machine without them; [dry-run mode](#dry-run-no-hardware) exercises the
> software loop with a simulated ONIX and synthetic frames. The code is published
> for transparency/reuse; the figures in the root README are fully reproducible
> **without** it.

## Launch

```bash
cd CLOSED_LOOP_GUI
python LaunchWebGUI.py            # serves http://127.0.0.1:5000
HOST=0.0.0.0 python LaunchWebGUI.py   # all interfaces; see Security below
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

**Save and Start.** Start saves the form first and does not launch if the save
fails. The server checks every field's type and range and rejects bad values
with a message; `num_channels` must be 2, because the ONIX templates
(`NN`/`AN`/`NA`/`AA`) cover exactly two channels. Keys the form does not show
(`watch_dir`, `experiment_templates`, `decision_key`, the log/setpoint/luminosity
paths, `dry_run`) keep their current `config.json` values. If you change
`global_path`, paths that sat under the old root move under the new one, and
paths set elsewhere stay as they are. `config.json` is written atomically.

## Files

| File | Role | Invocation |
|------|------|------------|
| `LaunchWebGUI.py` | **Entry point.** Flask "Closed-Loop Bio-Control Hub" web GUI on `127.0.0.1:5000` (set `HOST` to change). Bootstraps `config.json`, serves the dashboard, and exposes the `/api/*` endpoints (config, setpoints, frames/masks, segmentation, luminosity plot, log tail, pipeline start/stop). Start spawns `run_system.sh` and records its process group in `pipeline.pid`; Stop sends it SIGTERM and waits up to 75 s before SIGKILL. | `python LaunchWebGUI.py` |
| `templates/index.html` | Single-page dashboard served at `/`. | (served by `LaunchWebGUI.py`) |
| `static/app.js`, `static/style.css` | Browser client — polls the `/api/*` endpoints, renders frames/masks/plots and the readiness strip — plus styles. | (served by `LaunchWebGUI.py`) |
| `run_system.sh` | Orchestrator. Runs `config.py`, then launches the monitor daemons in parallel (plus `fake_frames.py` in dry-run mode). If any daemon exits, it stops the others. On Stop, SIGTERM or Ctrl-C it gives `SendDecisions.py` up to 60 s to abort and close the ONIX run, then SIGKILLs whatever is left. | `bash run_system.sh` (or via the GUI's Start button) |
| `config.py` | Config factory. `build_config()` derives every data path from a single `global_path` and holds all run knobs; `save_config()` creates the directories and writes `config.json` atomically. Run directly, it writes a default `config.json` only when none exists; otherwise it only creates the directories the existing file names. | `python config.py` |
| `config.json` | Resolved run config written next to the scripts on launch. **Git-ignored** (embeds `global_path` + the ONIX endpoint). | (generated) |
| `HandleSegmentations.py` | Cellpose (GPU) segmentation daemon. Polls `watch_dir`, writes `{frame:05d}_channel{ch}.npy` masks + `_meta.json` ROI counts. Launched **only** when `continuous_segmentation=True`. | (launched by `run_system.sh`) |
| `CreateDecisions.py` | Decision stage. Reads frames + masks, computes per-cell mean fluorescence, compares to per-channel setpoints, writes per-channel decisions and an atomic `final_decisions/actions.toml`. | (launched by `run_system.sh`) |
| `SendDecisions.py` | Actuation stage. Consumes `actions.toml`, manages per-channel pulse timers, maps channel state → ONIX experiment (`NN`/`AN`/`NA`/`AA`), and drives the ONIX server over HTTP (or a simulated ONIX when `dry_run` is true). Writes `media_status.json`, which the web GUI reads, and a timestamped hardware-telemetry CSV for offline inspection. | (launched by `run_system.sh`) |
| `MonitorPerformance.py` | Watches ROI-count metadata for per-channel cell gains/losses and logs a `FLAG:` line when a change reaches `threshold_ratio`. Every `cleanup_interval_sec` it moves files older than `retention_time_hours` from `directories_to_clean` into a dated zip under `global_path`, keeping the newest mask per channel so the GUI's mask preview and Push mask still work. | (launched by `run_system.sh`) |
| `fake_frames.py` | Dry-run frame source. Writes synthetic two-channel PNG frames into `watch_dir` every 5 s and a matching frame-0 mask per channel into `mask_dir`. Refuses to run unless `dry_run` is true. | (launched by `run_system.sh` when `dry_run=True`) |
| `io_utils.py` | Shared helpers (`log`, `load_config`, `parse_filename`). Imported by the scripts above. `parse_filename()` is the one frame filter: `channel_<c>..._timepoint_<t>` with a `.png`, `.jpg` or `.jpeg` extension, any case. | — |

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
media_status.json               (read by the web GUI)
ONIX_Hardware_Log_<ts>.csv      (telemetry, for offline inspection)
```

Stage 2 reads `monitoring.log` and `luminosity_log_channel<N>.json` (written by
`CreateDecisions.py`), not the files above.

All working directories (`watch_dir`, `processed_masks`, `current_masks`,
`temp_decisions`, `final_decisions`, …) live under `global_path` and are
created automatically by `config.py`.

## Configuration

Everything is driven by `config.py` → `config.json`. Edit `config.py` (or the run
config in the web UI) before first use. Key knobs:

- `global_path` — root for all data directories (**machine-specific**).
- `watch_dir` — directory of incoming frames. The default points at one example
  folder under `global_path`; set yours in `config.json` (a GUI Save keeps it).
- `num_channels` (must be 2), `threshold_ratio`, `acidic_pulse_sec`,
  `run_duration_sec`, `continuous_segmentation`.
- `dry_run` — simulated ONIX plus synthetic frames; see below. Not on the GUI
  form, so set it in `config.json`.
- `onix_server_ip` / `onix_server_port` — the ONIX2 hardware server endpoint.
- `experiment_templates` — Windows paths to the `.OnixExp` templates on the ONIX
  control PC.
- `decision_key` — media-action → integer mapping.

> **Note:** the shipped defaults in `config.py` / `templates/index.html`
> (`global_path`, `onix_server_ip`/`port`, `experiment_templates`) are
> non-identifying placeholders (e.g. the `192.0.2.10` documentation IP). Replace
> them with your own machine's values before first use.

## How the loop behaves

**Setpoints.** Once the frame-0 masks are pushed, `CreateDecisions.py` sets each
channel's setpoint to 200% of the mean frame-0 fluorescence inside that
channel's mask, and writes it to `setpoints.txt`. That write replaces the whole
file, so setpoints typed into the GUI before this moment are lost. Edits made
after it take effect on the next frame, because the file is re-read every
frame. A frame whose masked mean is at or above the setpoint is an "add acidic
media" decision. If a channel's frame-0 image cannot be read, the channel gets
no setpoint and is skipped, with a log line each frame, until you enter one in
the GUI.

**Acid pulses and their timing.** An acid decision on an idle channel starts
that channel's pulse timer in `SendDecisions.py`. The timer counts
`acidic_pulse_sec` from the moment `SendDecisions.py` reads the decision, not
from when the ONIX starts the new template. The two channels' pulse states pick
the template (`NN`, `AN`, `NA`, `AA`). Each change aborts and closes the running
ONIX experiment, then creates, opens, checks and starts the next one, and that
ONIX setup takes a few seconds. The setup time comes out of the pulse: the acid
template starts a few seconds after the decision and runs for slightly less
than `acidic_pulse_sec`. Acid decisions during a pulse are ignored; after it
expires, the next acid decision starts a new one.

**Run length.** One ONIX run lasts at most `run_duration_sec`, 86400 s or 24 h
by default, whatever the template, and a state change ends it sooner. When the
time runs out, the run is aborted and saved, and the same state starts again
about 5 s later. The pipeline never stops by itself.

**Failures.** If an ONIX run fails to start, `SendDecisions.py` retries after
5 s, doubling the wait on each further failure up to 5 min; a successful run
resets it. A retry of the same state reopens the experiment file that a failed
attempt created but never started, instead of making a new one.

**Stopping.** Stop, Ctrl-C and SIGTERM all make `SendDecisions.py` abort the
running ONIX experiment and close it, saved, before it exits. With a run live,
the log shows `Aborting run...` and `Closing and saving...` after
`>>> Shutting down pipeline... <<<`. If any daemon dies, `run_system.sh` shuts
the rest down the same way. A GUI restarted while an earlier pipeline is still
running finds it through `pipeline.pid`, refuses to start a second one, and can
still Stop it.

## Dry run (no hardware)

Set `"dry_run": true` in `config.json`, point `global_path` (or `watch_dir`) at a
scratch location, and leave `continuous_segmentation` false unless you have a
GPU. After Start, `SendDecisions.py` answers every ONIX command locally and
sends no network requests, and `run_system.sh` also starts `fake_frames.py`.
`fake_frames.py` writes synthetic frames whose brightness crosses the setpoint,
plus a frame-0 mask per channel in `mask_dir`. On the Segmentation tab, push the
frame-0 mask for channel 1 and channel 2 (no need to run Cellpose). The log then
shows decisions, acid pulses and `NN`/`AN`/`NA`/`AA` switching.

## ⚠️ Security / hardware

- `SendDecisions.py` issues HTTP requests that **create and run experiments on
  networked lab hardware** (perfusion + microscopy). Point it only at hardware you
  control, on a trusted network.
- `LaunchWebGUI.py` has **no authentication** and can Start/Stop the
  hardware-driving pipeline. It binds `127.0.0.1:5000`, so reach it through an
  SSH tunnel. `HOST=0.0.0.0` makes it listen on every interface; only do that on
  a trusted network. Loopback still lets any other user on the same machine
  reach it.
- Every POST must carry a JSON body and an `Origin` (or `Referer`) header that
  matches the server's own host and port, or the server answers 403. A page on
  another site cannot send such a request, which blocks cross-site request
  forgery from a browser that has the tunnel open.
- Start/Stop manage `run_system.sh` as a POSIX process group (`os.setsid` /
  `killpg`).

## Runtime artifacts

`config.json` is generated next to the scripts on launch and is **git-ignored**
(it embeds the resolved `global_path` and the ONIX endpoint) — do not commit it.
`pipeline.pid` sits next to it while a pipeline started from the GUI is running.
Acquired frames, masks, logs, and telemetry CSVs are written under `global_path`
and are not redistributed.

## Notes

- **Run from this directory.** The daemons load `config.json` via a relative path;
  `run_system.sh` and the web server `cd`/resolve here automatically. `io_utils.py`
  is imported as a bare module — keep these files co-located and **do not** put
  this directory on the same `sys.path` as `SCRIPTS/core_pipeline/` (each has its
  *own* `io_utils.py`).
- **GPU required** for segmentation: `CUDA_VISIBLE_DEVICES=0` and
  `CellposeModel(gpu=True)`.
- **POSIX-only** process management (`os.setsid` / `killpg`) in the launcher.
- Default mode is `continuous_segmentation=False` (frame-0 masks pushed from
  the GUI's Segmentation tab); set it `True` to segment every frame via
  `HandleSegmentations.py`.
