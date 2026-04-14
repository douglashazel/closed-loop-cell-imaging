"""
Napari-based GUI for the Closed-Loop Bio-Control Pipeline.

Launch with:  python LaunchNapari.py
No pre-existing config.json is needed -- this script runs config.py on startup.
"""

import atexit
import os
import sys
import signal
import subprocess
import threading
import time
import re
from collections import deque

import numpy as np
import napari
from magicgui.widgets import SpinBox, FloatSpinBox, Container
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QFrame, QScrollArea, QLineEdit, QSpinBox as QtSpinBox,
    QDoubleSpinBox, QCheckBox, QFileDialog,
)
from PIL import Image

from io_utils import load_config, parse_filename, log
from config import build_config, save_config


# ---------------------------------------------------------------------------
# Bootstrap: run config.py so config.json + directories exist
# ---------------------------------------------------------------------------
_V4_DIR = os.path.dirname(os.path.abspath(__file__))

def _ensure_config():
    """Create config.json with defaults if it does not already exist."""
    config_json = os.path.join(_V4_DIR, "config.json")
    if not os.path.exists(config_json):
        log("config.json not found -- creating with defaults...")
        cfg = build_config()
        save_config(cfg, _V4_DIR)
        log(f"config.json created at {config_json}")

_ensure_config()


# ---------------------------------------------------------------------------
# Qt style helpers
# ---------------------------------------------------------------------------
def _styled_button(text, color, text_color="#ffffff"):
    """Create a QPushButton with a colored background."""
    btn = QPushButton(text)
    btn.setStyleSheet(
        f"QPushButton {{"
        f"  background-color: {color};"
        f"  color: {text_color};"
        f"  font-weight: bold;"
        f"  font-size: 13px;"
        f"  padding: 8px 12px;"
        f"  border: none;"
        f"  border-radius: 4px;"
        f"}}"
        f"QPushButton:hover {{"
        f"  background-color: {color}; opacity: 0.85;"
        f"}}"
        f"QPushButton:pressed {{"
        f"  background-color: {color}; opacity: 0.7;"
        f"}}"
    )
    btn.setMinimumHeight(36)
    return btn


def _section_label(text):
    """Create a bold section header label."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-weight: bold; font-size: 13px; color: #cccccc;"
        "padding: 6px 0 2px 0;"
    )
    return lbl


def _status_label(text="Ready"):
    """Create a status display label."""
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "font-size: 12px; color: #aaaaaa; padding: 4px 6px;"
        "background-color: #2a2a2a; border-radius: 3px;"
    )
    lbl.setWordWrap(True)
    return lbl


def _separator():
    """Horizontal line separator."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #555555;")
    return line


def _config_row(label_text, widget, hint_text=None):
    """Create a labeled config field with an optional description underneath."""
    container = QWidget()
    vbox = QVBoxLayout(container)
    vbox.setContentsMargins(0, 2, 0, 6)
    vbox.setSpacing(2)
    lbl = QLabel(label_text)
    lbl.setStyleSheet("font-size: 12px; color: #cccccc;")
    vbox.addWidget(lbl)
    vbox.addWidget(widget)
    if hint_text:
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #666666; padding-left: 2px;")
        vbox.addWidget(hint)
    return container


_INPUT_MAX_WIDTH = 180


def _int_spinbox(value, lo, hi):
    """Create a Qt QSpinBox with dark styling."""
    sb = QtSpinBox()
    sb.setRange(lo, hi)
    sb.setValue(value)
    sb.setMaximumWidth(_INPUT_MAX_WIDTH)
    sb.setStyleSheet(
        "QSpinBox { background: #2a2a2a; color: #dddddd; border: 1px solid #555;"
        " border-radius: 3px; padding: 3px; font-size: 12px; }"
    )
    return sb


def _float_spinbox(value, lo, hi, decimals=3, step=0.01):
    """Create a Qt QDoubleSpinBox with dark styling."""
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    sb.setMaximumWidth(_INPUT_MAX_WIDTH)
    sb.setStyleSheet(
        "QDoubleSpinBox { background: #2a2a2a; color: #dddddd; border: 1px solid #555;"
        " border-radius: 3px; padding: 3px; font-size: 12px; }"
    )
    return sb


def _line_edit(text):
    """Create a styled QLineEdit."""
    le = QLineEdit(text)
    le.setMaximumWidth(_INPUT_MAX_WIDTH)
    le.setStyleSheet(
        "QLineEdit { background: #2a2a2a; color: #dddddd; border: 1px solid #555;"
        " border-radius: 3px; padding: 4px; font-size: 12px; }"
    )
    return le


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_FILENAME_RE = re.compile(r"channel_(\d+).*timepoint_(\d+)\.png$", re.IGNORECASE)


def _discover_channels(watch_dir):
    """Return sorted list of channel ints found in watch_dir for frame 0."""
    channels = set()
    if not os.path.isdir(watch_dir):
        return []
    for f in os.listdir(watch_dir):
        m = _FILENAME_RE.search(f)
        if m and int(m.group(2)) == 0:
            channels.add(int(m.group(1)))
    return sorted(channels)


def _latest_frame(watch_dir):
    """Return the highest frame number found in watch_dir, or 0."""
    best = 0
    if not os.path.isdir(watch_dir):
        return 0
    for f in os.listdir(watch_dir):
        m = _FILENAME_RE.search(f)
        if m:
            best = max(best, int(m.group(2)))
    return best


def _image_path(watch_dir, channel, frame):
    """Find the image file for a given channel and frame."""
    if not os.path.isdir(watch_dir):
        return None
    for f in sorted(os.listdir(watch_dir)):
        m = _FILENAME_RE.search(f)
        if m and int(m.group(1)) == channel and int(m.group(2)) == frame:
            return os.path.join(watch_dir, f)
    return None


def _mask_path(mask_dir, channel, frame):
    """Canonical mask path."""
    return os.path.join(mask_dir, f"{frame:05d}_channel{channel}.npy")


def _load_image(path):
    """Load an image as numpy array, returning None on failure."""
    try:
        return np.array(Image.open(path))
    except Exception:
        return None


def _load_mask(path):
    """Load a .npy mask, returning None on failure."""
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main Hub
# ---------------------------------------------------------------------------
class ClosedLoopHub:
    def __init__(self):
        self.cfg = load_config(os.path.join(_V4_DIR, "config.json"))
        self.viewer = napari.Viewer(title="Closed-Loop Bio-Control Hub")
        self.process = None

        # Napari layers (placeholder until real data arrives)
        self.image_layer = self.viewer.add_image(
            np.zeros((512, 512), dtype=np.uint16),
            name="Latest Raw Frame", colormap="gray",
        )
        self.mask_layer = self.viewer.add_labels(
            np.zeros((512, 512), dtype=np.int32),
            name="Current Segmentation",
        )
        # Shapes layer for user-drawn ROI (use Napari toolbar to draw)
        self.roi_layer = self.viewer.add_shapes(
            name="ROI",
            edge_color="yellow",
            edge_width=3,
            face_color="transparent",
        )

        # Tracking for live-view
        self._last_img_file = None
        self._last_mask_file = None

        # Log buffer for the log viewer widget
        self._log_lines = deque(maxlen=500)
        self._log_pos = 0  # file read position

        # --- Build the tabbed control panel ---
        self._tab_widget = QTabWidget()

        # Tab 1: Pipeline Controls
        self._build_pipeline_tab()
        # Tab 2: Segmentation / Preprocessing
        self._build_segmentation_tab()
        # Tab 3: Log Viewer
        self._build_log_tab()

        self.viewer.window.add_dock_widget(
            self._tab_widget, area="right", name="Controls"
        )

        # Polling timer: update layers + log
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(1000)

        # Ensure child processes are killed when napari closes or script exits
        self.viewer.window._qt_window.destroyed.connect(self.stop_pipeline)
        atexit.register(self.stop_pipeline)

    # -----------------------------------------------------------------------
    # Tab 1 -- Pipeline controls + configuration
    # -----------------------------------------------------------------------
    def _build_pipeline_tab(self):
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Pipeline controls ---
        layout.addWidget(_section_label("Pipeline Controls"))

        self._pipeline_status_label = _status_label("Pipeline: not running")
        layout.addWidget(self._pipeline_status_label)

        layout.addWidget(_separator())

        btn_start = _styled_button("START PIPELINE", "#2e7d32")  # green
        btn_stop = _styled_button("STOP PIPELINE", "#c62828")    # red

        btn_start.clicked.connect(self.start_pipeline)
        btn_stop.clicked.connect(self.stop_pipeline)

        layout.addWidget(btn_start)
        layout.addWidget(btn_stop)

        layout.addWidget(_separator())

        # --- Data path ---
        layout.addWidget(_section_label("Data Path"))

        self._path_display = QLabel(self.cfg.get("global_path", ""))
        self._path_display.setWordWrap(True)
        self._path_display.setStyleSheet(
            "font-size: 11px; color: #aaaaaa; padding: 6px;"
            "background-color: #2a2a2a; border: 1px solid #555;"
            "border-radius: 3px;"
        )
        layout.addWidget(self._path_display)

        btn_browse = _styled_button("Browse Data Directory...", "#455a64")
        btn_browse.clicked.connect(self._browse_global_path)
        layout.addWidget(btn_browse)

        layout.addWidget(_separator())

        # --- Configuration fields ---
        layout.addWidget(_section_label("Run Configuration"))
        save_hint = QLabel(
            "These settings are written to config.json when you click "
            "Save or Start. Restart the pipeline for changes to take effect."
        )
        save_hint.setWordWrap(True)
        save_hint.setStyleSheet("font-size: 10px; color: #777777; padding-bottom: 4px;")
        layout.addWidget(save_hint)

        cfg = self.cfg

        # -- Imaging --
        self._cfg_num_channels = _int_spinbox(cfg.get("num_channels", 1), 1, 10)
        layout.addWidget(_config_row(
            "Number of Channels", self._cfg_num_channels,
            "How many imaging channels to process. Each channel gets its "
            "own segmentation mask and independent decision.",
        ))

        self._cfg_continuous_seg = QCheckBox("Enable continuous segmentation")
        self._cfg_continuous_seg.setChecked(cfg.get("continuous_segmentation", False))
        self._cfg_continuous_seg.setStyleSheet("color: #cccccc; font-size: 12px;")
        layout.addWidget(_config_row(
            "Continuous Segmentation", self._cfg_continuous_seg,
            "If checked, every incoming frame is segmented (archived to "
            "mask_dir). If unchecked, only frame 0 is segmented. "
            "Decisions always use the initial frame-0 mask from curr_mask_dir.",
        ))

        layout.addWidget(_separator())

        # -- Decision logic --
        self._cfg_threshold_ratio = _float_spinbox(
            cfg.get("threshold_ratio", 0.05), 0.0, 1.0, decimals=3, step=0.005,
        )
        layout.addWidget(_config_row(
            "Threshold Ratio", self._cfg_threshold_ratio,
            "Intensity threshold for media-switching decisions. A cell's "
            "mean pixel value is compared against the setpoint +/- this "
            "ratio to decide between neutral, acidic, or basic media.",
        ))

        layout.addWidget(_separator())

        # -- File polling --
        self._cfg_num_tries = _int_spinbox(cfg.get("num_tries", 10), 1, 200)
        layout.addWidget(_config_row(
            "Num Tries", self._cfg_num_tries,
            "Max number of attempts to read a newly arrived image file "
            "before skipping it (handles partially-written files).",
        ))

        self._cfg_sleep_time = _float_spinbox(
            cfg.get("sleep_time", 0.2), 0.01, 30.0, decimals=2, step=0.1,
        )
        layout.addWidget(_config_row(
            "Sleep Time (sec)", self._cfg_sleep_time,
            "Delay in seconds between each file-polling attempt.",
        ))

        layout.addWidget(_separator())

        # -- Timing --
        self._cfg_run_duration = _int_spinbox(
            cfg.get("run_duration_sec", 86400), 1, 604800,
        )
        layout.addWidget(_config_row(
            "Run Duration (sec)", self._cfg_run_duration,
            "Duration of the neutral media run in seconds. "
            "86400 = 24 hours. Pipeline runs until this expires or is stopped.",
        ))

        self._cfg_acidic_pulse = _int_spinbox(
            cfg.get("acidic_pulse_sec", 30), 1, 7200,
        )
        layout.addWidget(_config_row(
            "Acidic Pulse Duration (sec)", self._cfg_acidic_pulse,
            "How long each acidic media pulse lasts before automatically "
            "switching back to neutral media.",
        ))

        layout.addWidget(_separator())

        # -- ONIX connection --
        layout.addWidget(_section_label("ONIX Connection"))

        self._cfg_onix_ip = _line_edit(cfg.get("onix_server_ip", "192.0.2.10"))
        layout.addWidget(_config_row(
            "ONIX Server IP", self._cfg_onix_ip,
            "IP address of the ONIX2 hardware controller.",
        ))

        self._cfg_onix_port = _int_spinbox(
            cfg.get("onix_server_port", 8881), 1, 65535,
        )
        layout.addWidget(_config_row(
            "ONIX Server Port", self._cfg_onix_port,
            "TCP port for the ONIX2 server connection.",
        ))

        layout.addWidget(_separator())

        # -- Cleanup --
        layout.addWidget(_section_label("File Cleanup"))

        self._cfg_retention_hours = _int_spinbox(
            cfg.get("retention_time_hours", 2), 0, 720,
        )
        layout.addWidget(_config_row(
            "Retention Time (hours)", self._cfg_retention_hours,
            "How long to keep processed masks and overlays before they "
            "are cleaned up. Set to 0 to disable cleanup.",
        ))

        self._cfg_cleanup_interval = _int_spinbox(
            cfg.get("cleanup_interval_sec", 1800), 60, 86400,
        )
        layout.addWidget(_config_row(
            "Cleanup Interval (sec)", self._cfg_cleanup_interval,
            "How often the system checks for old files to clean up. "
            "1800 = every 30 minutes.",
        ))

        layout.addWidget(_separator())

        # -- Save button --
        btn_save = _styled_button("Save Configuration", "#1565c0")  # blue
        btn_save.clicked.connect(self._save_config)
        layout.addWidget(btn_save)

        self._config_status_label = _status_label("")
        layout.addWidget(self._config_status_label)

        layout.addStretch()

        # Wrap in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        self._tab_widget.addTab(scroll, "Pipeline")

    # -----------------------------------------------------------------------
    # Tab 2 -- Segmentation & mask management
    # -----------------------------------------------------------------------
    def _build_segmentation_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Cellpose parameters ---
        layout.addWidget(_section_label("Cellpose Segmentation"))

        self._seg_channel = SpinBox(value=1, min=1, max=6, label="Channel")
        self._seg_frame = SpinBox(value=0, min=0, max=99999, label="Frame")
        self._seg_diameter = SpinBox(value=50, min=1, max=500, label="Diameter")
        self._seg_flow = FloatSpinBox(
            value=10.0, min=0.0, max=100.0, step=0.5, label="Flow threshold",
        )
        self._seg_cellprob = FloatSpinBox(
            value=0.1, min=-10.0, max=10.0, step=0.1, label="Cellprob threshold",
        )
        self._seg_niter = SpinBox(value=600, min=0, max=5000, label="Niter (0=default)")

        params = Container(widgets=[
            self._seg_channel, self._seg_frame,
            self._seg_diameter, self._seg_flow, self._seg_cellprob, self._seg_niter,
        ])
        params.native.setMaximumWidth(_INPUT_MAX_WIDTH * 2)
        layout.addWidget(params.native)

        btn_segment = _styled_button("Run Cellpose Segmentation", "#1565c0")  # blue
        btn_segment.clicked.connect(self._run_segmentation)
        layout.addWidget(btn_segment)

        self._seg_status_label = _status_label("Ready")
        layout.addWidget(self._seg_status_label)

        layout.addWidget(_separator())

        # --- ROI controls ---
        layout.addWidget(_section_label("ROI Filter"))
        roi_hint = QLabel("Draw a shape on the ROI layer, then accept it")
        roi_hint.setStyleSheet("font-size: 11px; color: #888888; padding: 0 0 4px 0;")
        roi_hint.setWordWrap(True)
        layout.addWidget(roi_hint)

        btn_apply_roi = _styled_button("Accept Drawn ROI", "#e65100")  # orange
        btn_apply_roi.clicked.connect(self._apply_roi)
        layout.addWidget(btn_apply_roi)

        layout.addWidget(_separator())

        # --- Update Masks (push to curr_mask_dir) ---
        layout.addWidget(_section_label("Mask Management"))

        btn_update = _styled_button("Update Masks (Start Decisions)", "#2e7d32")  # green
        btn_update.clicked.connect(self._update_masks)
        layout.addWidget(btn_update)

        layout.addWidget(_separator())

        # --- Visualize existing mask ---
        layout.addWidget(_section_label("Visualization"))

        btn_load_raw = _styled_button("Load Frame (Raw + Mask)", "#455a64")  # gray-blue
        btn_load_raw.clicked.connect(self._load_frame)
        layout.addWidget(btn_load_raw)

        layout.addStretch()
        self._tab_widget.addTab(tab, "Segmentation")

    # -----------------------------------------------------------------------
    # Tab 3 -- Log viewer
    # -----------------------------------------------------------------------
    def _build_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Live media indicator (one bar per channel) ---
        layout.addWidget(_section_label("Channel Media Status"))

        num_ch = self.cfg.get("num_channels", 2)
        self._ch_media_labels = {}
        _idle_style = (
            "font-size: 13px; font-weight: bold; color: #aaaaaa;"
            "padding: 8px 10px; background-color: #2a2a2a;"
            "border: 1px solid #555; border-radius: 4px;"
        )
        for ch in range(1, num_ch + 1):
            lbl = QLabel(f"CH{ch}: IDLE")
            lbl.setStyleSheet(_idle_style)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)
            self._ch_media_labels[ch] = lbl

        # Summary label showing combined ONIX experiment name (NN/AN/NA/AA)
        self._experiment_label = QLabel("")
        self._experiment_label.setStyleSheet(
            "font-size: 11px; color: #777777; padding: 2px 6px;"
        )
        layout.addWidget(self._experiment_label)

        layout.addWidget(_separator())

        layout.addWidget(_section_label("Live Log  (monitoring.log)"))

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(
            "QTextEdit {"
            "  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;"
            "  font-size: 11px;"
            "  background-color: #1a1a1a;"
            "  color: #d4d4d4;"
            "  border: 1px solid #444444;"
            "  border-radius: 3px;"
            "  padding: 4px;"
            "}"
        )
        self._log_text.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self._log_text)

        btn_clear = _styled_button("Clear Log Display", "#455a64")
        btn_clear.clicked.connect(self._clear_log_display)
        layout.addWidget(btn_clear)

        self._tab_widget.addTab(tab, "Log")

    # -----------------------------------------------------------------------
    # Polling (runs every 1 s on the Qt event loop)
    # -----------------------------------------------------------------------
    def _poll(self):
        self._update_layers()
        self._update_log()
        self._update_pipeline_status()
        self._update_media_status()

    def _update_layers(self):
        """Load newest raw image and mask from disk into the viewer."""
        watch_dir = self.cfg["watch_dir"]
        mask_dir = self.cfg["mask_dir"]

        try:
            if os.path.isdir(watch_dir):
                images = sorted(
                    f for f in os.listdir(watch_dir)
                    if f.lower().endswith(('.png', '.jpg'))
                )
                if images and images[-1] != self._last_img_file:
                    self._last_img_file = images[-1]
                    img = _load_image(os.path.join(watch_dir, images[-1]))
                    if img is not None:
                        self.image_layer.data = img

            if os.path.isdir(mask_dir):
                masks = sorted(
                    f for f in os.listdir(mask_dir) if f.endswith('.npy')
                )
                if masks and masks[-1] != self._last_mask_file:
                    self._last_mask_file = masks[-1]
                    msk = _load_mask(os.path.join(mask_dir, masks[-1]))
                    if msk is not None:
                        self.mask_layer.data = msk
        except Exception:
            pass

    def _update_log(self):
        """Tail the monitoring.log file and append new lines to the log widget."""
        log_path = os.path.join(self.cfg["global_path"], "monitoring.log")
        if not os.path.isfile(log_path):
            return
        try:
            with open(log_path, "r") as f:
                f.seek(self._log_pos)
                new_text = f.read()
                self._log_pos = f.tell()
            if new_text:
                self._log_text.moveCursor(
                    self._log_text.textCursor().End
                )
                self._log_text.insertPlainText(new_text)
                self._log_text.moveCursor(
                    self._log_text.textCursor().End
                )
        except Exception:
            pass

    def _update_pipeline_status(self):
        lbl = self._pipeline_status_label
        if self.process is not None and self.process.poll() is None:
            lbl.setText("Pipeline: RUNNING  (pid {})".format(self.process.pid))
            lbl.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #a5d6a7;"
                "padding: 4px 6px; background-color: #1b3a1b; border-radius: 3px;"
            )
        elif self.process is not None and self.process.poll() is not None:
            code = self.process.poll()
            lbl.setText("Pipeline: exited  (code {})".format(code))
            lbl.setStyleSheet(
                "font-size: 12px; color: #ef9a9a;"
                "padding: 4px 6px; background-color: #3a1b1b; border-radius: 3px;"
            )
        else:
            lbl.setText("Pipeline: not running")
            lbl.setStyleSheet(
                "font-size: 12px; color: #aaaaaa;"
                "padding: 4px 6px; background-color: #2a2a2a; border-radius: 3px;"
            )

    def _update_media_status(self):
        """Read media_status.json and update the per-channel indicators."""
        import json
        status_path = os.path.join(self.cfg["final_dir"], "media_status.json")
        try:
            with open(status_path, "r") as f:
                data = json.load(f)
        except Exception:
            return  # file missing or unreadable — leave labels as-is

        channels = data.get("channels", {})
        pulse_duration = data.get("pulse_duration", 30)
        experiment = data.get("experiment", "")

        _neutral_style = (
            "font-size: 13px; font-weight: bold; color: #a5d6a7;"
            "padding: 8px 10px; background-color: #1b3a1b;"
            "border: 1px solid #2e7d32; border-radius: 4px;"
        )
        _acidic_style = (
            "font-size: 13px; font-weight: bold; color: #ffcdd2;"
            "padding: 8px 10px; background-color: #4a1a1a;"
            "border: 1px solid #c62828; border-radius: 4px;"
        )
        _idle_style = (
            "font-size: 13px; font-weight: bold; color: #aaaaaa;"
            "padding: 8px 10px; background-color: #2a2a2a;"
            "border: 1px solid #555; border-radius: 4px;"
        )

        for ch_str, ch_data in channels.items():
            ch = int(ch_str)
            if ch not in self._ch_media_labels:
                continue
            lbl = self._ch_media_labels[ch]
            state = ch_data.get("state", "neutral")
            pulse_start = ch_data.get("pulse_start")

            if state == "acidic" and pulse_start is not None:
                remaining = max(0, pulse_duration - (time.time() - pulse_start))
                secs = int(remaining)
                if remaining <= 0:
                    text = f"CH{ch}: ACIDIC — pulse complete, switching..."
                else:
                    text = f"CH{ch}: ACIDIC PULSE — {secs}s remaining"
                lbl.setText(text)
                lbl.setStyleSheet(_acidic_style)
            else:
                lbl.setText(f"CH{ch}: NEUTRAL")
                lbl.setStyleSheet(_neutral_style)

        if experiment:
            self._experiment_label.setText(f"ONIX experiment: {experiment}")

    # -----------------------------------------------------------------------
    # Configuration actions
    # -----------------------------------------------------------------------
    def _browse_global_path(self):
        """Open a directory picker for the global data path."""
        current = self._path_display.text() or os.path.expanduser("~")
        chosen = QFileDialog.getExistingDirectory(
            None, "Select Data Directory", current,
        )
        if chosen:
            self._path_display.setText(chosen)
            self._config_status_label.setText(
                "Path updated -- click Save to apply"
            )

    def _save_config(self):
        """Rebuild config from the UI widgets and save to config.json."""
        global_path = self._path_display.text()
        overrides = {
            "num_channels": self._cfg_num_channels.value(),
            "threshold_ratio": self._cfg_threshold_ratio.value(),
            "num_tries": self._cfg_num_tries.value(),
            "sleep_time": self._cfg_sleep_time.value(),
            "onix_server_ip": self._cfg_onix_ip.text().strip(),
            "onix_server_port": self._cfg_onix_port.value(),
            "retention_time_hours": self._cfg_retention_hours.value(),
            "cleanup_interval_sec": self._cfg_cleanup_interval.value(),
            "run_duration_sec": self._cfg_run_duration.value(),
            "acidic_pulse_sec": self._cfg_acidic_pulse.value(),
            "continuous_segmentation": self._cfg_continuous_seg.isChecked(),
        }
        cfg = build_config(global_path, **overrides)
        saved = save_config(cfg, _V4_DIR)
        self.cfg = cfg
        self._config_status_label.setText("Saved to {}".format(saved))
        log(f"Configuration saved: global_path={global_path}")

    # -----------------------------------------------------------------------
    # Pipeline control actions
    # -----------------------------------------------------------------------
    def start_pipeline(self):
        if self.process is not None and self.process.poll() is None:
            log("Pipeline is already running.")
            return
        # Save current UI config so the pipeline uses the latest values
        self._save_config()
        log("Launching run_system.sh from Napari...")
        self.process = subprocess.Popen(
            ["bash", "./run_system.sh"],
            cwd=_V4_DIR,
            preexec_fn=os.setsid,
        )

    def stop_pipeline(self):
        if self.process is None or self.process.poll() is not None:
            log("No pipeline process to stop.")
            return
        log("Shutting down pipeline processes...")
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.process = None

    # -----------------------------------------------------------------------
    # Segmentation actions
    # -----------------------------------------------------------------------
    def _run_segmentation(self):
        """Run Cellpose on a single frame/channel with the GUI parameters."""
        channel = self._seg_channel.value
        frame = self._seg_frame.value
        watch_dir = self.cfg["watch_dir"]
        mask_dir = self.cfg["mask_dir"]
        temp_overlays = self.cfg["temp_overlays"]

        img_p = _image_path(watch_dir, channel, frame)
        if img_p is None:
            self._seg_status_label.setText("No image found for ch {} frame {}".format(
                channel, frame
            ))
            return

        self._seg_status_label.setText("Segmenting...")
        # Run in a thread so the GUI stays responsive
        threading.Thread(
            target=self._segmentation_worker,
            args=(img_p, channel, frame, mask_dir, temp_overlays),
            daemon=True,
        ).start()

    def _segmentation_worker(self, img_path, channel, frame, mask_dir, temp_overlays):
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            from cellpose import models, io, utils
            from scipy.ndimage import binary_dilation

            model = models.CellposeModel(gpu=True)
            img = io.imread(img_path)

            eval_kwargs = {"diameter": self._seg_diameter.value}
            flow = self._seg_flow.value
            if flow > 0:
                eval_kwargs["flow_threshold"] = flow
            cellprob = self._seg_cellprob.value
            eval_kwargs["cellprob_threshold"] = cellprob
            niter = self._seg_niter.value
            if niter > 0:
                eval_kwargs["niter"] = niter

            masks, _, _ = model.eval(img, **eval_kwargs)

            base = f"{frame:05d}_channel{channel}"
            np.save(os.path.join(mask_dir, base + ".npy"), masks)

            # Save overlay image
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
            fig.savefig(
                os.path.join(temp_overlays, base + "_overlay.png"),
                dpi=300, bbox_inches="tight",
            )
            plt.close(fig)

            # Update viewer layers from the main thread via the next poll,
            # but also set them directly for immediate feedback
            num_cells = len(np.unique(masks)) - 1
            self.image_layer.data = img
            self.mask_layer.data = masks

            self._seg_status_label.setText("Done: {} cells detected".format(num_cells))
            log(f"Segmentation complete: ch{channel} frame {frame} -- {num_cells} cells")

        except Exception as e:
            self._seg_status_label.setText("Error: {}".format(e))
            log(f"Segmentation error: {e}")

    # -----------------------------------------------------------------------
    # ROI layer lookup (resilient to deletion / re-creation)
    # -----------------------------------------------------------------------
    def _get_roi_layer(self):
        """
        Return the Shapes layer named 'ROI'.  If it was deleted, create a
        fresh one so the user can keep working.
        """
        try:
            layer = self.viewer.layers["ROI"]
            if layer.__class__.__name__ == "Shapes":
                self.roi_layer = layer
                return layer
        except KeyError:
            pass
        # Layer is missing — recreate it
        self.roi_layer = self.viewer.add_shapes(
            name="ROI",
            edge_color="yellow",
            edge_width=3,
            face_color="transparent",
        )
        return self.roi_layer

    # -----------------------------------------------------------------------
    # ROI filter (uses shapes drawn on the "ROI" layer)
    # -----------------------------------------------------------------------
    def _apply_roi(self):
        """
        Rasterize all shapes on the ROI layer into a binary mask.
        Cells whose centroid falls outside the combined ROI are zeroed out.
        The filtered mask is saved back to mask_dir.
        """
        roi_layer = self._get_roi_layer()

        channel = self._seg_channel.value
        frame = self._seg_frame.value
        mask_dir = self.cfg["mask_dir"]

        mpath = _mask_path(mask_dir, channel, frame)
        mask = _load_mask(mpath)
        if mask is None:
            self._seg_status_label.setText("No mask at {}".format(mpath))
            return

        if len(roi_layer.data) == 0:
            self._seg_status_label.setText("Draw a shape on the ROI layer first")
            return

        h, w = mask.shape[:2]

        # Rasterize shapes into a binary (row, col) mask
        roi_binary = roi_layer.to_masks(mask_shape=(h, w))
        # Union all drawn shapes into one combined ROI
        roi_combined = np.any(roi_binary, axis=0)

        # Find cells and their centroids
        cell_ids = np.unique(mask)
        cell_ids = cell_ids[cell_ids != 0]
        if len(cell_ids) == 0:
            self._seg_status_label.setText("Mask has no cells")
            return

        # Keep cells whose centroid is inside the drawn ROI
        keep = []
        for cid in cell_ids:
            ys, xs = np.where(mask == cid)
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            # Clamp to image bounds
            cy = min(max(cy, 0), h - 1)
            cx = min(max(cx, 0), w - 1)
            if roi_combined[cy, cx]:
                keep.append(cid)

        keep = np.array(keep)
        filtered = np.where(np.isin(mask, keep), mask, 0)

        np.save(mpath, filtered)
        self.mask_layer.data = filtered

        n_kept = len(keep)
        n_removed = len(cell_ids) - n_kept
        self._seg_status_label.setText("ROI: kept {} cells, removed {}".format(
            n_kept, n_removed
        ))
        log(f"ROI filter applied: ch{channel} frame {frame} -- "
            f"{n_kept} kept, {n_removed} removed")

    # -----------------------------------------------------------------------
    # Update Masks (push to curr_mask_dir -- starts decision loop)
    # -----------------------------------------------------------------------
    def _update_masks(self):
        """
        Copy the mask for the selected channel/frame from mask_dir to
        curr_mask_dir. This is the signal that unblocks CreateDecisions.py.
        """
        import shutil
        channel = self._seg_channel.value
        frame = self._seg_frame.value
        mask_dir = self.cfg["mask_dir"]
        curr_mask_dir = self.cfg["curr_mask_dir"]

        src = _mask_path(mask_dir, channel, frame)
        if not os.path.exists(src):
            self._seg_status_label.setText("No mask found: {}".format(src))
            return

        # Remove any old masks for this channel in curr_mask_dir
        for f in os.listdir(curr_mask_dir):
            if f.endswith(f"_channel{channel}.npy"):
                os.remove(os.path.join(curr_mask_dir, f))

        dst = os.path.join(curr_mask_dir, f"{frame:05d}_channel{channel}.npy")
        shutil.copy2(src, dst)

        self._seg_status_label.setText("Mask pushed: {}".format(os.path.basename(dst)))
        log(f"Updated channel{channel} mask -> {dst}")

    # -----------------------------------------------------------------------
    # Load frame (raw + mask) into viewer
    # -----------------------------------------------------------------------
    def _load_frame(self):
        """Load a specific frame/channel into the viewer layers."""
        channel = self._seg_channel.value
        frame = self._seg_frame.value
        watch_dir = self.cfg["watch_dir"]
        mask_dir = self.cfg["mask_dir"]

        img_p = _image_path(watch_dir, channel, frame)
        if img_p is not None:
            img = _load_image(img_p)
            if img is not None:
                self.image_layer.data = img

        mpath = _mask_path(mask_dir, channel, frame)
        mask = _load_mask(mpath)
        if mask is not None:
            self.mask_layer.data = mask

        has_img = img_p is not None
        has_mask = os.path.exists(mpath)
        self._seg_status_label.setText("Loaded ch{} frame {} (img={}, mask={})".format(
            channel, frame,
            "yes" if has_img else "no",
            "yes" if has_mask else "no",
        ))

    # -----------------------------------------------------------------------
    # Log display helpers
    # -----------------------------------------------------------------------
    def _clear_log_display(self):
        self._log_text.clear()


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hub = ClosedLoopHub()
    napari.run()
