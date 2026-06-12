"""
Flask-based web GUI for the Closed-Loop Bio-Control Pipeline.

Replaces LaunchNapari.py. Serves a single-page HTML dashboard on port 5000
with the same Pipeline / Segmentation / Log controls. Intended for access
via SSH port-forward:

    ssh -L 5000:localhost:5000 user@ssh-host
    # then browse http://localhost:5000 on your local machine

Launch with:  python LaunchWebGUI.py
"""

import atexit
import io
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, render_template, send_file, abort
from PIL import Image

from io_utils import load_config, log
from config import build_config, save_config

# ---------------------------------------------------------------------------
# Bootstrap: ensure config.json exists before we serve anything
# ---------------------------------------------------------------------------
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_APP_DIR, "config.json")


def _ensure_config():
    if not os.path.exists(_CONFIG_PATH):
        log("config.json not found -- creating with defaults...")
        cfg = build_config()
        save_config(cfg, _APP_DIR)
        log(f"config.json created at {_CONFIG_PATH}")


_ensure_config()


# ---------------------------------------------------------------------------
# Global runtime state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
pipeline_proc = None  # subprocess.Popen | None
seg_thread = None     # threading.Thread | None
seg_status = {"state": "idle", "message": "Ready", "frame": 0, "channel": 1}

_FILENAME_RE = re.compile(r"channel_(\d+).*timepoint_(\d+)\.png$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers (file lookup, image/mask loading) — lifted from LaunchNapari
# ---------------------------------------------------------------------------
def _load_cfg():
    return load_config(_CONFIG_PATH)


def _image_path(watch_dir, channel, frame):
    if not os.path.isdir(watch_dir):
        return None
    for f in sorted(os.listdir(watch_dir)):
        m = _FILENAME_RE.search(f)
        if m and int(m.group(1)) == channel and int(m.group(2)) == frame:
            return os.path.join(watch_dir, f)
    return None


def _mask_path(mask_dir, channel, frame):
    return os.path.join(mask_dir, f"{frame:05d}_channel{channel}.npy")


def _load_image_np(path):
    try:
        return np.array(Image.open(path))
    except Exception:
        return None


def _load_mask_np(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


def _array_to_png_bytes(arr, mode="L"):
    """Encode a numpy array as a PNG bytes object."""
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _normalize_gray(img):
    """Scale any numeric image to 8-bit grayscale for browser display."""
    if img is None:
        return None
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


def _colorize_labels(mask):
    """Map a labels array to an RGBA image: background transparent, each
    cell a distinct hue. Used for the mask overlay in the browser."""
    if mask is None:
        return None
    mask = mask.astype(np.int32)
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return out
    rng = np.random.default_rng(42)
    lut = rng.integers(64, 255, size=(int(ids.max()) + 1, 3), dtype=np.uint8)
    for cid in ids:
        sel = mask == cid
        out[sel, 0:3] = lut[cid]
        out[sel, 3] = 140  # semi-transparent
    return out


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(_APP_DIR, "templates"),
    static_folder=os.path.join(_APP_DIR, "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


# ---- Config ---------------------------------------------------------------
@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(_load_cfg())


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Rebuild config from posted fields and save to config.json.

    Mirrors LaunchNapari._save_config: uses build_config() to regenerate
    derived paths when global_path changes, then save_config() to ensure
    directories and write the file atomically via the stdlib pattern."""
    body = request.get_json(force=True) or {}
    global_path = body.get("global_path") or _load_cfg().get("global_path")

    overrides = {}
    for key in (
        "num_channels",
        "threshold_ratio",
        "num_tries",
        "sleep_time",
        "onix_server_ip",
        "onix_server_port",
        "retention_time_hours",
        "cleanup_interval_sec",
        "run_duration_sec",
        "acidic_pulse_sec",
        "continuous_segmentation",
    ):
        if key in body:
            overrides[key] = body[key]

    cfg = build_config(global_path, **overrides)
    saved = save_config(cfg, _APP_DIR)
    log(f"Configuration saved via web GUI: global_path={global_path}")
    return jsonify({"ok": True, "saved_to": saved, "config": cfg})


# ---- Pipeline start/stop --------------------------------------------------
@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global pipeline_proc
    with _state_lock:
        if pipeline_proc is not None and pipeline_proc.poll() is None:
            return jsonify({"ok": False, "error": "already running",
                            "pid": pipeline_proc.pid})
        log("Launching run_system.sh from web GUI...")
        pipeline_proc = subprocess.Popen(
            ["bash", "./run_system.sh"],
            cwd=_APP_DIR,
            preexec_fn=os.setsid,
        )
        return jsonify({"ok": True, "pid": pipeline_proc.pid})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global pipeline_proc
    with _state_lock:
        if pipeline_proc is None or pipeline_proc.poll() is not None:
            return jsonify({"ok": False, "error": "not running"})
        log("Shutting down pipeline processes...")
        try:
            os.killpg(os.getpgid(pipeline_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Give it a moment to exit cleanly, then SIGKILL if stuck
        for _ in range(30):
            if pipeline_proc.poll() is not None:
                break
            time.sleep(0.1)
        else:
            try:
                os.killpg(os.getpgid(pipeline_proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        pipeline_proc = None
        return jsonify({"ok": True})


@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    global pipeline_proc
    if pipeline_proc is None:
        return jsonify({"running": False, "pid": None, "exit_code": None})
    rc = pipeline_proc.poll()
    if rc is None:
        return jsonify({"running": True, "pid": pipeline_proc.pid, "exit_code": None})
    return jsonify({"running": False, "pid": pipeline_proc.pid, "exit_code": rc})


# ---- System readiness -----------------------------------------------------
def _onix_reachable(ip, port, timeout=0.2):
    try:
        with socket.create_connection((ip, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _frame0_present(watch_dir, num_channels):
    """Return {ch: bool} for whether frame-0 image exists for each channel."""
    present = {ch: False for ch in range(1, num_channels + 1)}
    if not os.path.isdir(watch_dir):
        return present
    for f in os.listdir(watch_dir):
        m = _FILENAME_RE.search(f)
        if m and int(m.group(2)) == 0:
            ch = int(m.group(1))
            if ch in present:
                present[ch] = True
    return present


def _last_decision_record(luminosity_file, num_channels):
    """Find the most-recent decision across all per-channel luminosity logs."""
    base, ext = os.path.splitext(luminosity_file)
    latest_frame = -1
    latest_mtime = None
    for ch in range(1, num_channels + 1):
        path = f"{base}_channel{ch}{ext}"
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                records = json.load(f)
            if records:
                fr = records[-1].get("frame", -1)
                if fr > latest_frame:
                    latest_frame = fr
                    latest_mtime = os.path.getmtime(path)
        except (OSError, ValueError):
            continue
    age = None
    if latest_mtime is not None:
        age = max(0, time.time() - latest_mtime)
    return latest_frame, age


@app.route("/api/system/readiness", methods=["GET"])
def api_system_readiness():
    """Snapshot of operator-facing system state — drives the readiness strip
    and the per-channel mask chips. All values are cheap derivations of state
    the pipeline already produces; nothing is cached."""
    global pipeline_proc
    cfg = _load_cfg()
    num_channels = int(cfg.get("num_channels", 2))
    watch_dir = cfg.get("watch_dir", "")
    curr_mask_dir = cfg.get("curr_mask_dir", "")

    masks_ready = {}
    for ch in range(1, num_channels + 1):
        masks_ready[ch] = os.path.isfile(
            os.path.join(curr_mask_dir, f"00000_channel{ch}.npy")
        )

    frame0 = _frame0_present(watch_dir, num_channels)
    last_frame, last_age = _last_decision_record(
        cfg.get("luminosity_file", ""), num_channels
    )

    pipeline_running = (pipeline_proc is not None and pipeline_proc.poll() is None)

    return jsonify({
        "config_saved": os.path.isfile(_CONFIG_PATH),
        "watch_dir_exists": os.path.isdir(watch_dir),
        "frame0_present": frame0,
        "masks_ready": masks_ready,
        "onix_reachable": _onix_reachable(
            cfg.get("onix_server_ip", ""), cfg.get("onix_server_port", 0)
        ),
        "pipeline_running": pipeline_running,
        "last_decision_frame": last_frame if last_frame >= 0 else None,
        "last_decision_age_sec": last_age,
        "num_channels": num_channels,
    })


@app.route("/api/luminosity", methods=["GET"])
def api_luminosity():
    """Return the last N records of the per-channel luminosity log (default 80).

    Feeds the rail sparkline (pushLumi) without re-rendering the matplotlib PNG."""
    cfg = _load_cfg()
    try:
        channel = int(request.args.get("channel", 1))
        limit = int(request.args.get("limit", 80))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "bad channel/limit"}), 400
    base, ext = os.path.splitext(cfg["luminosity_file"])
    path = f"{base}_channel{channel}{ext}"
    if not os.path.isfile(path):
        return jsonify({"ok": True, "channel": channel, "records": []})
    try:
        with open(path) as f:
            records = json.load(f)
    except (OSError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({
        "ok": True,
        "channel": channel,
        "records": records[-limit:] if limit > 0 else records,
    })


# ---- Log tail -------------------------------------------------------------
@app.route("/api/log/tail", methods=["GET"])
def api_log_tail():
    """Seek monitoring.log from byte offset ?pos=N and return new text.

    Matches LaunchNapari._update_log semantics exactly: client keeps the
    last returned position and passes it back on the next poll."""
    cfg = _load_cfg()
    log_path = cfg.get("log_path") or os.path.join(cfg["global_path"], "monitoring.log")
    pos = int(request.args.get("pos", 0))
    if not os.path.isfile(log_path):
        return jsonify({"text": "", "pos": 0, "exists": False})
    try:
        size = os.path.getsize(log_path)
        if pos > size:
            pos = 0  # log was rotated/truncated
        with open(log_path, "r") as f:
            f.seek(pos)
            text = f.read()
            new_pos = f.tell()
        return jsonify({"text": text, "pos": new_pos, "exists": True})
    except Exception as e:
        return jsonify({"text": "", "pos": pos, "exists": True, "error": str(e)})


# ---- Setpoints ------------------------------------------------------------
@app.route("/api/setpoints", methods=["GET"])
def api_get_setpoints():
    """Read setpoints.txt and return {channel: value} for each num_channels.

    Missing file or missing channels return empty string so the UI can show
    a placeholder until CreateDecisions has computed the initial values."""
    cfg = _load_cfg()
    setpoint_file = cfg["setpoint_file"]
    num_channels = int(cfg.get("num_channels", 2))
    values = {}
    if os.path.isfile(setpoint_file):
        try:
            with open(setpoint_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("setpoint_channel") and "=" in line:
                        key, val = line.split("=", 1)
                        ch = int(key[len("setpoint_channel"):])
                        values[ch] = float(val)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    channels = [{"channel": ch, "value": values.get(ch)}
                for ch in range(1, num_channels + 1)]
    return jsonify({"ok": True, "channels": channels,
                    "path": setpoint_file, "exists": os.path.isfile(setpoint_file)})


@app.route("/api/setpoints", methods=["POST"])
def api_set_setpoints():
    """Update setpoints.txt from posted {channel: value} map.

    Merges with any existing values so partial updates don't wipe channels
    the user didn't touch. CreateDecisions.load_setpoints() re-reads the file
    on every frame, so changes take effect on the next decision."""
    body = request.get_json(force=True) or {}
    updates_in = body.get("channels") or {}
    try:
        updates = {int(k): float(v) for k, v in updates_in.items()
                   if v is not None and str(v).strip() != ""}
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"invalid value: {e}"}), 400

    cfg = _load_cfg()
    setpoint_file = cfg["setpoint_file"]

    current = {}
    if os.path.isfile(setpoint_file):
        try:
            with open(setpoint_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("setpoint_channel") and "=" in line:
                        key, val = line.split("=", 1)
                        current[int(key[len("setpoint_channel"):])] = float(val)
        except Exception as e:
            return jsonify({"ok": False, "error": f"read failed: {e}"}), 500

    current.update(updates)
    os.makedirs(os.path.dirname(setpoint_file), exist_ok=True)
    try:
        tmp = setpoint_file + ".tmp"
        with open(tmp, "w") as f:
            for ch in sorted(current):
                f.write(f"setpoint_channel{ch}={current[ch]:.6f}\n")
        os.rename(tmp, setpoint_file)
    except Exception as e:
        return jsonify({"ok": False, "error": f"write failed: {e}"}), 500

    log("Setpoints updated via web GUI: "
        + ", ".join(f"ch{ch}={current[ch]:.3f}" for ch in sorted(updates)))
    return jsonify({"ok": True, "channels": current})


# ---- Luminosity plot ------------------------------------------------------
_CHANNEL_COLORS = {1: "steelblue", 2: "seagreen"}
_PLOT_PARAMS = {
    "dpi": 150,
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "setpoint_color": "gray",
    "acid_color": "tomato",
}


@app.route("/api/luminosity-plot.png", methods=["GET"])
def api_luminosity_plot():
    """Render mean-luminosity-vs-frame plot across all per-channel JSON logs.

    Mirrors the notebook snippet: one subplot per channel, dashed setpoint
    line, dotted vertical lines marking 'add acidic media' frames."""
    cfg = _load_cfg()
    base, ext = os.path.splitext(cfg["luminosity_file"])
    pattern = re.compile(r"^" + re.escape(os.path.basename(base))
                         + r"_channel(\d+)" + re.escape(ext) + r"$")
    log_dir = os.path.dirname(base)

    by_channel = {}
    if os.path.isdir(log_dir):
        for fname in sorted(os.listdir(log_dir)):
            m = pattern.match(fname)
            if not m:
                continue
            try:
                with open(os.path.join(log_dir, fname)) as f:
                    by_channel[int(m.group(1))] = json.load(f)
            except Exception as e:
                log(f"luminosity plot: failed to read {fname}: {e}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = sorted(by_channel.keys())
    if not channels:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=_PLOT_PARAMS["dpi"])
        ax.text(0.5, 0.5, "No luminosity logs found yet",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        fig, axes = plt.subplots(len(channels), 1, dpi=_PLOT_PARAMS["dpi"],
                                 figsize=(8, 3 * len(channels)), sharex=True)
        if len(channels) == 1:
            axes = [axes]
        for ax, ch in zip(axes, channels):
            records = by_channel[ch]
            if not records:
                ax.text(0.5, 0.5, f"Channel {ch}: empty log",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            frames = [d["frame"] for d in records]
            luminosity = [d["mean_luminosity"] for d in records]
            setpoint = records[-1]["setpoint"]
            acid_frames = [d["frame"] for d in records
                           if d.get("decision") == "add acidic media"]

            ax.plot(frames, luminosity,
                    color=_CHANNEL_COLORS.get(ch, "black"),
                    linewidth=1.5, label=f"Channel {ch} Mean Luminosity")
            ax.axhline(setpoint, color=_PLOT_PARAMS["setpoint_color"],
                       linewidth=1, linestyle="--",
                       label=f"Setpoint ({setpoint})")
            for i, f in enumerate(acid_frames):
                ax.axvline(f, color=_PLOT_PARAMS["acid_color"],
                           linewidth=0.8, linestyle=":",
                           label="Acidic Pulse" if i == 0 else None)
            ax.set_title(f"Channel {ch}",
                         fontsize=_PLOT_PARAMS["title_fontsize"] - 4,
                         fontweight=_PLOT_PARAMS["title_fontweight"])
            ax.set_ylabel("Mean Luminosity")
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="lower left", fontsize=8)
        axes[-1].set_xlabel("Frame")
        fig.suptitle("Mean Luminosity Over Frames",
                     fontsize=_PLOT_PARAMS["title_fontsize"],
                     fontweight=_PLOT_PARAMS["title_fontweight"])
        fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ---- Media status ---------------------------------------------------------
@app.route("/api/media-status", methods=["GET"])
def api_media_status():
    cfg = _load_cfg()
    status_path = os.path.join(cfg["final_dir"], "media_status.json")
    try:
        with open(status_path, "r") as f:
            data = json.load(f)
        data["_server_time"] = time.time()
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"channels": {}, "pulse_duration": cfg.get("acidic_pulse_sec", 30),
                        "experiment": "", "_server_time": time.time()})
    except Exception as e:
        return jsonify({"error": str(e), "_server_time": time.time()}), 500


# ---- Segmentation ---------------------------------------------------------
@app.route("/api/segmentation/run", methods=["POST"])
def api_segmentation_run():
    global seg_thread, seg_status
    body = request.get_json(force=True) or {}
    channel = int(body.get("channel", 1))
    frame = int(body.get("frame", 0))
    diameter = int(body.get("diameter", 50))
    flow_threshold = float(body.get("flow_threshold", 10.0))
    cellprob_threshold = float(body.get("cellprob_threshold", 0.1))
    niter = int(body.get("niter", 600))

    with _state_lock:
        if seg_thread is not None and seg_thread.is_alive():
            return jsonify({"ok": False, "error": "segmentation already running"})

        cfg = _load_cfg()
        img_p = _image_path(cfg["watch_dir"], channel, frame)
        if img_p is None:
            return jsonify({"ok": False,
                            "error": f"No image found for ch {channel} frame {frame}"})

        seg_status = {"state": "running",
                      "message": "Segmenting...",
                      "frame": frame, "channel": channel}

        seg_thread = threading.Thread(
            target=_segmentation_worker,
            args=(img_p, channel, frame, cfg["mask_dir"], cfg["temp_overlays"],
                  diameter, flow_threshold, cellprob_threshold, niter),
            daemon=True,
        )
        seg_thread.start()
    return jsonify({"ok": True, "started": True})


_cached_cellpose_model = None
_cellpose_model_lock = threading.Lock()


def _get_cellpose_model():
    """Return a cached Cellpose model, loading it once on first use."""
    global _cached_cellpose_model
    with _cellpose_model_lock:
        if _cached_cellpose_model is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            from cellpose import models
            log("Loading Cellpose model (first use)...")
            _cached_cellpose_model = models.CellposeModel(gpu=True)
            log("Cellpose model ready.")
        return _cached_cellpose_model


def _segmentation_worker(img_path, channel, frame, mask_dir, temp_overlays,
                         diameter, flow_threshold, cellprob_threshold, niter):
    """Cellpose worker — lifted from LaunchNapari._segmentation_worker."""
    global seg_status
    try:
        from cellpose import io as cpio, utils
        from scipy.ndimage import binary_dilation

        model = _get_cellpose_model()
        img = cpio.imread(img_path)

        eval_kwargs = {"diameter": diameter}
        if flow_threshold > 0:
            eval_kwargs["flow_threshold"] = flow_threshold
        eval_kwargs["cellprob_threshold"] = cellprob_threshold
        if niter > 0:
            eval_kwargs["niter"] = niter

        masks, _, _ = model.eval(img, **eval_kwargs)

        base = f"{frame:05d}_channel{channel}"
        np.save(os.path.join(mask_dir, base + ".npy"), masks)

        outlines = utils.masks_to_outlines(masks)
        outlines = binary_dilation(outlines, iterations=3)
        overlay = img.copy()
        if overlay.ndim == 2:
            overlay = np.stack([overlay] * 3, axis=-1)
        overlay[outlines] = [255, 0, 0]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        axes[0].imshow(overlay)
        axes[0].set_title("With segmentation")
        axes[0].axis("off")
        axes[1].imshow(img, cmap="gray")
        axes[1].set_title("Raw image")
        axes[1].axis("off")
        fig.savefig(os.path.join(temp_overlays, base + "_overlay.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        num_cells = int(len(np.unique(masks)) - 1)
        seg_status = {"state": "done",
                      "message": f"Done: {num_cells} cells detected",
                      "frame": frame, "channel": channel, "num_cells": num_cells}
        log(f"Segmentation complete: ch{channel} frame {frame} -- {num_cells} cells")
    except Exception as e:
        seg_status = {"state": "error", "message": f"Error: {e}",
                      "frame": frame, "channel": channel}
        log(f"Segmentation error: {e}")


@app.route("/api/segmentation/status", methods=["GET"])
def api_segmentation_status():
    return jsonify(seg_status)


@app.route("/api/segmentation/update-masks", methods=["POST"])
def api_segmentation_update_masks():
    """Copy mask_dir/{frame}_channel{ch}.npy -> curr_mask_dir/00000_channel{ch}.npy.

    Mirrors LaunchNapari._update_masks — the signal that unblocks
    CreateDecisions.py."""
    body = request.get_json(force=True) or {}
    channel = int(body.get("channel", 1))
    frame = int(body.get("frame", 0))
    cfg = _load_cfg()
    mask_dir = cfg["mask_dir"]
    curr_mask_dir = cfg["curr_mask_dir"]

    src = _mask_path(mask_dir, channel, frame)
    if not os.path.exists(src):
        return jsonify({"ok": False, "error": f"No mask found: {src}"})

    os.makedirs(curr_mask_dir, exist_ok=True)
    for f in os.listdir(curr_mask_dir):
        if f.endswith(f"_channel{channel}.npy"):
            os.remove(os.path.join(curr_mask_dir, f))

    # CreateDecisions.py waits for 00000_channel{ch}.npy specifically; the source
    # frame is operator-chosen but the pushed file always represents the reference.
    dst = os.path.join(curr_mask_dir, f"00000_channel{channel}.npy")
    shutil.copy2(src, dst)
    log(f"pushed channel{channel} mask (from frame {frame}) -> {dst}")
    return jsonify({"ok": True, "dst": dst})


# ---- Frame / mask image serving ------------------------------------------
@app.route("/api/frames", methods=["GET"])
def api_frames():
    """Return sorted list of {frame, channels:[...]} available in watch_dir."""
    cfg = _load_cfg()
    watch_dir = cfg["watch_dir"]
    frames = {}
    if os.path.isdir(watch_dir):
        for f in os.listdir(watch_dir):
            m = _FILENAME_RE.search(f)
            if m:
                ch, fr = int(m.group(1)), int(m.group(2))
                frames.setdefault(fr, set()).add(ch)
    out = [{"frame": fr, "channels": sorted(list(chs))}
           for fr, chs in sorted(frames.items())]
    return jsonify({"frames": out})


@app.route("/api/frame/<int:frame>/<int:channel>.png", methods=["GET"])
def api_frame_png(frame, channel):
    cfg = _load_cfg()
    p = _image_path(cfg["watch_dir"], channel, frame)
    if p is None:
        abort(404)
    img = _load_image_np(p)
    if img is None:
        abort(500)
    gray = _normalize_gray(img)
    return send_file(_array_to_png_bytes(gray, mode="L"), mimetype="image/png")


@app.route("/api/mask/<int:frame>/<int:channel>.png", methods=["GET"])
def api_mask_png(frame, channel):
    cfg = _load_cfg()
    p = _mask_path(cfg["mask_dir"], channel, frame)
    if not os.path.exists(p):
        abort(404)
    mask = _load_mask_np(p)
    if mask is None:
        abort(500)
    rgba = _colorize_labels(mask)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ---------------------------------------------------------------------------
# Cleanup on exit: kill pipeline subprocess if still running
# ---------------------------------------------------------------------------
def _cleanup():
    global pipeline_proc
    if pipeline_proc is not None and pipeline_proc.poll() is None:
        log("Web GUI exiting -- killing pipeline subprocess")
        try:
            os.killpg(os.getpgid(pipeline_proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


atexit.register(_cleanup)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log("Starting web GUI on http://0.0.0.0:5000  (forward via: ssh -L 5000:localhost:5000 ...)")
    # threaded=True so log polls + segmentation don't block each other
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False, use_reloader=False)
