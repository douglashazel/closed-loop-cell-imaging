"""
Flask-based web GUI for the Closed-Loop Bio-Control Pipeline.

Serves a single-page HTML dashboard on port 5000
with the same Pipeline / Segmentation / Log controls. Binds 127.0.0.1 unless
the HOST environment variable says otherwise. Intended for access via SSH
port-forward:

    ssh -L 5000:localhost:5000 user@ssh-host
    # then browse http://localhost:5000 on your local machine

Launch with:  python LaunchWebGUI.py
"""

import atexit
import io
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

import numpy as np
from flask import Flask, jsonify, request, render_template, send_file, abort
from PIL import Image

from io_utils import load_config, log, parse_filename
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

# Process-group id of the running pipeline, kept on disk so a restarted GUI
# still knows about a pipeline an earlier GUI process started
_PIDFILE = os.path.join(_APP_DIR, "pipeline.pid")
# run_system.sh gives SendDecisions up to 60 s to abort and close the ONIX
# run on Stop; wait a little longer than that before SIGKILLing the group
_STOP_GRACE_SEC = 75


def _read_pidfile():
    try:
        with open(_PIDFILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _remove_pidfile():
    try:
        os.remove(_PIDFILE)
    except FileNotFoundError:
        pass


def _pipeline_group_alive(pgid):
    """True while process group *pgid* is alive and, if its leader is still
    there, that leader is run_system.sh (not an unrelated process reusing the pid)."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        with open(f"/proc/{pgid}/cmdline", "rb") as f:
            return b"run_system.sh" in f.read()
    except OSError:
        return True  # leader already gone (or no /proc) but the group lives on


def _running_pgid():
    """Process-group id of the live pipeline, or None.

    Covers the pipeline this GUI started and one left behind by an earlier
    GUI process, so Start never launches a second pipeline on the same ONIX."""
    if pipeline_proc is not None and pipeline_proc.poll() is None:
        return pipeline_proc.pid
    pgid = _read_pidfile()
    if pgid is not None and _pipeline_group_alive(pgid):
        return pgid
    return None


def _check_leftover_pipeline():
    pgid = _read_pidfile()
    if pgid is None:
        return
    if _pipeline_group_alive(pgid):
        log(f"Pipeline started by an earlier GUI is still running (process group "
            f"{pgid}); Start is refused until it is stopped (Stop button or "
            f"kill -TERM -{pgid})")
    else:
        _remove_pidfile()


_check_leftover_pipeline()


# ---------------------------------------------------------------------------
# Helpers (file lookup, image/mask loading)
# ---------------------------------------------------------------------------
def _load_cfg():
    return load_config(_CONFIG_PATH)


def _image_path(watch_dir, channel, frame):
    if not os.path.isdir(watch_dir):
        return None
    for f in sorted(os.listdir(watch_dir)):
        if parse_filename(f) == (channel, frame):
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
        return np.load(path)  # plain int label array; np.load refuses pickles by default
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


@app.before_request
def _reject_cross_site_writes():
    """CSRF guard: a state-changing request must carry a JSON body and come
    from a page served by this server (Origin header, or Referer if the
    browser sent no Origin). A form or script on another site cannot do both."""
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    source = request.headers.get("Origin") or request.headers.get("Referer") or ""
    if not request.is_json or urlparse(source).netloc.lower() != request.host.lower():
        log(f"Rejected {request.method} {request.path}: not a same-origin JSON request")
        return jsonify({"ok": False,
                        "error": "forbidden: only same-origin JSON requests are accepted"}), 403
    return None


def _json_object():
    """The request's JSON body if it is an object, else None."""
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else None


def _bad_body():
    return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400


@app.route("/")
def index():
    return render_template("index.html")


# ---- Config ---------------------------------------------------------------
@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(_load_cfg())


# Fields a Save may set: (type, min, max), None = unbounded. The ranges match
# the inputs in templates/index.html. Every other key keeps its config.json value.
_CONFIG_FIELDS = {
    "num_channels":            (int,   2, 2),
    "threshold_ratio":         (float, 0, 1),
    "num_tries":               (int,   1, 200),
    "sleep_time":              (float, 0.01, 30),
    "onix_server_ip":          (str,   None, None),
    "onix_server_port":        (int,   1, 65535),
    "retention_time_hours":    (int,   0, 720),
    "cleanup_interval_sec":    (int,   60, 86400),
    "run_duration_sec":        (int,   1, None),
    "acidic_pulse_sec":        (int,   1, 7200),
    "continuous_segmentation": (bool,  None, None),
    "dry_run":                 (bool,  None, None),
}


def _coerce_field(key, value):
    """Return *value* as the type config field *key* needs; ValueError if it can't be."""
    kind, lo, hi = _CONFIG_FIELDS[key]
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"{key} must be true or false, got {value!r}")
    if kind is str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")
        return value.strip()
    try:
        if isinstance(value, bool):
            raise TypeError
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}")
    if not math.isfinite(num):
        raise ValueError(f"{key} must be a finite number, got {value!r}")
    if kind is int:
        if not num.is_integer():
            raise ValueError(f"{key} must be a whole number, got {value!r}")
        num = int(num)
    if key == "num_channels" and num != 2:
        raise ValueError("num_channels must be 2: the ONIX experiment templates "
                         "(NN/AN/NA/AA) cover exactly two channels")
    if (lo is not None and num < lo) or (hi is not None and num > hi):
        bound = f"between {lo} and {hi}" if hi is not None else f"at least {lo}"
        raise ValueError(f"{key} must be {bound}, got {num}")
    return num


def _rebase(value, old_root, new_root):
    """Move a path (or list of paths) under old_root to the same place under new_root."""
    if isinstance(value, list):
        return [_rebase(v, old_root, new_root) for v in value]
    old = old_root.rstrip("/")
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new_root.rstrip("/") + value[len(old):]
    return value


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Validate posted fields and save config.json.

    Keys the body does not set keep their current config.json value (so
    watch_dir, experiment_templates, decision_key, dry_run, ... survive a
    Save); when global_path changes, paths under the old root move to the
    new one. Keys config.json lacks fall back to config.py defaults.
    save_config() creates the directories and writes the file atomically."""
    body = _json_object()
    if body is None:
        return _bad_body()
    current = _load_cfg()
    old_root = current.get("global_path")
    global_path = body.get("global_path") or old_root
    if not isinstance(global_path, str) or not os.path.isabs(global_path.strip()):
        return jsonify({"ok": False,
                        "error": f"global_path must be an absolute path, got {global_path!r}"}), 400
    global_path = global_path.strip()
    try:
        updates = {key: _coerce_field(key, body[key])
                   for key in _CONFIG_FIELDS if key in body}
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    cfg = build_config(global_path)
    for key in cfg:
        if key == "global_path" or key not in current:
            continue
        value = current[key]
        if old_root and global_path != old_root:
            value = _rebase(value, old_root, global_path)
        cfg[key] = value
    cfg.update(updates)
    saved = save_config(cfg, _APP_DIR)
    log(f"Configuration saved via web GUI: global_path={global_path}")
    return jsonify({"ok": True, "saved_to": saved, "config": cfg})


# ---- Pipeline start/stop --------------------------------------------------
@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global pipeline_proc
    with _state_lock:
        pgid = _running_pgid()
        if pgid is not None:
            return jsonify({"ok": False, "error": "already running", "pid": pgid})
        log("Launching run_system.sh from web GUI...")
        pipeline_proc = subprocess.Popen(
            ["bash", "./run_system.sh"],
            cwd=_APP_DIR,
            preexec_fn=os.setsid,
        )
        # setsid makes run_system.sh the leader of a new group: pgid == pid
        with open(_PIDFILE, "w") as f:
            f.write(f"{pipeline_proc.pid}\n")
        return jsonify({"ok": True, "pid": pipeline_proc.pid})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global pipeline_proc
    with _state_lock:
        pgid = _running_pgid()
        if pgid is None:
            return jsonify({"ok": False, "error": "not running"})
        log(f"Shutting down pipeline processes (waiting up to {_STOP_GRACE_SEC} s "
            "for SendDecisions to abort and close the ONIX run)...")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # run_system.sh waits for its daemons, then exits; SIGKILL if stuck
        deadline = time.time() + _STOP_GRACE_SEC
        while time.time() < deadline:
            if pipeline_proc is not None:
                pipeline_proc.poll()  # reap run_system.sh once it exits
            if not _pipeline_group_alive(pgid):
                break
            time.sleep(0.2)
        else:
            log(f"Pipeline still running after {_STOP_GRACE_SEC} s -- sending SIGKILL")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        pipeline_proc = None
        _remove_pidfile()
        return jsonify({"ok": True})


@app.route("/api/pipeline/status", methods=["GET"])
def api_pipeline_status():
    pgid = _running_pgid()
    if pgid is not None:
        return jsonify({"running": True, "pid": pgid, "exit_code": None})
    if pipeline_proc is None:
        return jsonify({"running": False, "pid": None, "exit_code": None})
    return jsonify({"running": False, "pid": pipeline_proc.pid,
                    "exit_code": pipeline_proc.returncode})


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
        ch, frame = parse_filename(f)
        if frame == 0 and ch in present:
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

    pipeline_running = _running_pgid() is not None

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

    Cclient keeps the last returned position and passes it back on the next poll."""
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
    body = _json_object()
    if body is None:
        return _bad_body()
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

    # Figure (Agg) rather than pyplot: pyplot's global state is not thread-safe
    from matplotlib.figure import Figure

    channels = sorted(by_channel.keys())
    if not channels:
        fig = Figure(figsize=(8, 3), dpi=_PLOT_PARAMS["dpi"])
        ax = fig.subplots()
        ax.text(0.5, 0.5, "No luminosity logs found yet",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        fig = Figure(figsize=(8, 3 * len(channels)), dpi=_PLOT_PARAMS["dpi"])
        axes = fig.subplots(len(channels), 1, sharex=True)
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
            ax.legend(loc="lower left", fontsize=8);
        axes[-1].set_xlabel("Frame")
        fig.suptitle("Mean Luminosity Over Frames",
                     fontsize=_PLOT_PARAMS["title_fontsize"],
                     fontweight=_PLOT_PARAMS["title_fontweight"])
        fig.tight_layout();

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
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
    body = _json_object()
    if body is None:
        return _bad_body()
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
    """Cellpose worker"""
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

        from matplotlib.figure import Figure
        fig = Figure(figsize=(10, 5), dpi=300)
        axes = fig.subplots(1, 2)
        axes[0].imshow(overlay)
        axes[0].set_title("With segmentation")
        axes[0].axis("off")
        axes[1].imshow(img, cmap="gray")
        axes[1].set_title("Raw image")
        axes[1].axis("off")
        fig.savefig(os.path.join(temp_overlays, base + "_overlay.png"),
                    dpi=300, bbox_inches="tight")

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
    """Copy mask_dir/{frame}_channel{ch}.npy -> curr_mask_dir/00000_channel{ch}.npy."""
    body = _json_object()
    if body is None:
        return _bad_body()
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
            ch, fr = parse_filename(f)
            if ch is not None:
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
    # Loopback only by default. HOST=0.0.0.0 exposes this unauthenticated GUI,
    # which can start the hardware pipeline, to the whole network.
    host = os.environ.get("HOST", "127.0.0.1")
    log(f"Starting web GUI on http://{host}:5000  (forward via: ssh -L 5000:localhost:5000 ...)")
    # threaded=True so log polls + segmentation don't block each other
    app.run(host=host, port=5000, threaded=True, debug=False, use_reloader=False)
