"""
napari-based preprocessing GUI for the cell analysis pipeline.

Launch:  python preprocess_gui.py
"""

import os
import re
import sys
import random
import subprocess
from dataclasses import dataclass, field

import numpy as np
import napari
from napari.qt import thread_worker
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QFileDialog,
    QTextEdit, QComboBox, QCheckBox, QProgressBar, QMessageBox,
    QApplication, QGroupBox,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from scipy.spatial import Delaunay
from scipy.ndimage import center_of_mass
from PIL import Image as PILImage

SPIN_MAX_WIDTH = 120  # keep spinboxes compact in the dock panel


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions (copied from notebook / scripts to avoid import side-effects)
# ═══════════════════════════════════════════════════════════════════════════════

def timepoint_sort_key(fname):
    m = re.search(r'timepoint_(\d+)', fname)
    return int(m.group(1)) if m else float('inf')


def load_segmentation(path):
    seg = np.load(path, allow_pickle=True)
    if isinstance(seg, dict):
        return seg['masks']
    try:
        return seg.item()['masks']
    except Exception:
        return seg


def get_delaunay_neighbors(cell_id, centroids_by_id):
    ids = np.array(list(centroids_by_id.keys()))
    coords = np.array([centroids_by_id[i] for i in ids])
    chosen_pos = np.where(ids == cell_id)[0][0]
    tri = Delaunay(coords)
    neighbors = set()
    for simplex in tri.simplices:
        if chosen_pos in simplex:
            neighbors.update(simplex)
    neighbors.discard(chosen_pos)
    return [int(ids[i]) for i in neighbors]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared pipeline state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineState:
    # Paths
    global_dir: str = ""
    frames_dir: str = ""
    masks_dir: str = ""
    save_path: str = ""
    all_frames: list = field(default_factory=list)
    all_masks: list = field(default_factory=list)

    # Section 0 – Cellpose
    flow_threshold: float = 0.975
    cellprob_threshold: float = -4.0
    niter: int = 3000
    diameter: int = 26
    frame_idx: int = 0
    temp_segmentation: np.ndarray = None
    cellpose_model: object = None  # cached CellposeModel

    # Section 1 – Save interval
    save_interval: int = 10

    # Section 2 – Shift
    shift_frame_idx: int = 1
    shift_xy: tuple = (0, 0)

    # Section 3 – Max distance
    max_distance: float = 30.0

    # Section 4 – ROI
    radius: int = 2000
    y_shift: int = 0
    x_shift: int = 0

    # Section 6 – Duplicate
    grace_period: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Layer helper
# ═══════════════════════════════════════════════════════════════════════════════

def _set_layer(viewer, data, name, layer_type="image", **kwargs):
    """Add or update a named layer, returning it."""
    try:
        layer = viewer.layers[name]
        layer.data = data
        for k, v in kwargs.items():
            if hasattr(layer, k):
                setattr(layer, k, v)
        return layer
    except KeyError:
        adder = getattr(viewer, f"add_{layer_type}")
        return adder(data, name=name, **kwargs)


def _remove_layer(viewer, name):
    try:
        viewer.layers.remove(name)
    except (KeyError, ValueError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 0 – Experiment Selection + Cellpose Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

class CellposeTab(QWidget):
    def __init__(self, viewer, state, on_experiment_loaded=None):
        super().__init__()
        self.viewer = viewer
        self.state = state
        self.on_experiment_loaded = on_experiment_loaded

        layout = QVBoxLayout(self)

        # ── Experiment selector ───────────────────────────────────────────
        exp_group = QGroupBox("Experiment")
        exp_layout = QVBoxLayout(exp_group)

        self.btn_select = QPushButton("Select Experiment Directory")
        self.btn_select.clicked.connect(self._select_experiment)
        exp_layout.addWidget(self.btn_select)

        self.lbl_exp = QLabel("No experiment selected")
        self.lbl_exp.setWordWrap(True)
        exp_layout.addWidget(self.lbl_exp)

        layout.addWidget(exp_group)

        # ── Frame index ───────────────────────────────────────────────────
        frame_group = QGroupBox("Preview Frame")
        frame_layout = QFormLayout(frame_group)

        self.spin_frame = QSpinBox()
        self.spin_frame.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_frame.setMinimum(0)
        self.spin_frame.setMaximum(0)
        self.spin_frame.valueChanged.connect(self._on_frame_changed)
        frame_layout.addRow("Frame index:", self.spin_frame)

        layout.addWidget(frame_group)

        # ── Cellpose parameters ───────────────────────────────────────────
        cp_group = QGroupBox("Cellpose Parameters")
        cp_layout = QFormLayout(cp_group)

        self.spin_flow = QDoubleSpinBox()
        self.spin_flow.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_flow.setRange(-10.0, 10.0)
        self.spin_flow.setSingleStep(0.025)
        self.spin_flow.setDecimals(3)
        self.spin_flow.setValue(self.state.flow_threshold)
        self.spin_flow.setToolTip("Lower = stricter (fewer false positives)")
        cp_layout.addRow("flow_threshold:", self.spin_flow)

        self.spin_cellprob = QDoubleSpinBox()
        self.spin_cellprob.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_cellprob.setRange(-10.0, 10.0)
        self.spin_cellprob.setSingleStep(0.5)
        self.spin_cellprob.setDecimals(1)
        self.spin_cellprob.setValue(self.state.cellprob_threshold)
        self.spin_cellprob.setToolTip("Lower = include lower-confidence cells")
        cp_layout.addRow("cellprob_threshold:", self.spin_cellprob)

        self.spin_niter = QSpinBox()
        self.spin_niter.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_niter.setRange(1, 10000)
        self.spin_niter.setSingleStep(100)
        self.spin_niter.setValue(self.state.niter)
        self.spin_niter.setToolTip("More iterations = more accurate, slower")
        cp_layout.addRow("niter:", self.spin_niter)

        self.spin_diam = QSpinBox()
        self.spin_diam.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_diam.setRange(0, 500)
        self.spin_diam.setValue(self.state.diameter)
        self.spin_diam.setToolTip("Approximate cell diameter in pixels (0 = auto)")
        cp_layout.addRow("diameter:", self.spin_diam)

        layout.addWidget(cp_group)

        # ── Run button + status ───────────────────────────────────────────
        self.btn_run = QPushButton("Run Cellpose Preview")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_cellpose)
        layout.addWidget(self.btn_run)

        self.lbl_status = QLabel("")
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    # ── Experiment selection ──────────────────────────────────────────────
    def _select_experiment(self):
        start = os.path.join(os.path.dirname(__file__), "EXPERIMENTS")
        if not os.path.isdir(start):
            start = os.path.dirname(__file__)
        path = QFileDialog.getExistingDirectory(self, "Select Experiment Directory", start)
        if not path:
            return

        frames_dir = os.path.join(path, "frames")
        masks_dir = os.path.join(path, "masks")
        if not os.path.isdir(frames_dir):
            QMessageBox.warning(self, "Invalid directory",
                                f"No 'frames/' subdirectory found in:\n{path}")
            return

        self.state.global_dir = path
        self.state.frames_dir = frames_dir
        self.state.masks_dir = masks_dir
        self.state.save_path = os.path.join(path, "analysis")
        self.state.all_frames = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))],
            key=timepoint_sort_key,
        )
        self.state.all_masks = sorted(
            [f for f in os.listdir(masks_dir) if f.endswith('.npy')],
            key=timepoint_sort_key,
        ) if os.path.isdir(masks_dir) else []

        n_frames = len(self.state.all_frames)
        n_masks = len(self.state.all_masks)
        rel = os.path.relpath(path, os.path.dirname(__file__))
        self.lbl_exp.setText(f"{rel}\nFrames: {n_frames}  |  Masks: {n_masks}")

        self.spin_frame.setMaximum(max(0, n_frames - 1))
        self.spin_frame.setValue(0)
        self.state.frame_idx = 0
        self.btn_run.setEnabled(n_frames > 0)

        # Load first frame into viewer
        if self.state.all_frames:
            self._load_frame_preview()

        if self.on_experiment_loaded:
            self.on_experiment_loaded()

    def _on_frame_changed(self, val):
        self.state.frame_idx = val
        if self.state.all_frames:
            self._load_frame_preview()

    def _load_frame_preview(self):
        img_path = os.path.join(self.state.frames_dir,
                                self.state.all_frames[self.state.frame_idx])
        img = np.array(PILImage.open(img_path))
        _set_layer(self.viewer, img, "Frame Preview", layer_type="image", colormap="gray")

    # ── Cellpose ─────────────────────────────────────────────────────────
    def _run_cellpose(self):
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Running Cellpose...")

        img_path = os.path.join(self.state.frames_dir,
                                self.state.all_frames[self.state.frame_idx])
        img = np.array(PILImage.open(img_path))

        ft = self.spin_flow.value()
        cp = self.spin_cellprob.value()
        ni = self.spin_niter.value()
        di = self.spin_diam.value()

        # Store params on state
        self.state.flow_threshold = ft
        self.state.cellprob_threshold = cp
        self.state.niter = ni
        self.state.diameter = di

        cached_model = self.state.cellpose_model

        @thread_worker
        def _work():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            from cellpose import models
            if cached_model is None:
                model = models.CellposeModel(gpu=True)
            else:
                model = cached_model
            masks, _, _ = model.eval(
                [img], flow_threshold=ft, cellprob_threshold=cp,
                niter=ni, diameter=di if di > 0 else None,
            )
            return model, masks[0]

        worker = _work()
        worker.returned.connect(self._on_cellpose_done)
        worker.start()

    def _on_cellpose_done(self, result):
        model, masks = result
        self.state.cellpose_model = model
        self.state.temp_segmentation = masks
        n_cells = int(masks.max()) if masks.max() > 0 else 0
        self.lbl_status.setText(f"Detected {n_cells} cells")
        self.btn_run.setEnabled(True)

        _set_layer(self.viewer, masks, "Cellpose Segmentation", layer_type="labels")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 – Save Interval
# ═══════════════════════════════════════════════════════════════════════════════

class SaveIntervalTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)

        group = QGroupBox("Save Interval")
        form = QFormLayout(group)

        self.lbl_auto = QLabel("Auto: --")
        form.addRow("Recommended:", self.lbl_auto)

        self.spin_interval = QSpinBox()
        self.spin_interval.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_interval.setRange(1, 10000)
        self.spin_interval.setValue(self.state.save_interval)
        self.spin_interval.setToolTip("How often (in frames) to save intermediate results")
        self.spin_interval.valueChanged.connect(self._on_changed)
        form.addRow("save_interval:", self.spin_interval)

        layout.addWidget(group)
        layout.addStretch()

    def refresh(self):
        n = len(self.state.all_frames)
        auto = 10 if n <= 500 else 100
        self.lbl_auto.setText(f"{auto}  ({n} frames)")
        self.spin_interval.setValue(auto)
        self.state.save_interval = auto

    def _on_changed(self, val):
        self.state.save_interval = val


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 – Frame Shift
# ═══════════════════════════════════════════════════════════════════════════════

class ShiftTab(QWidget):
    def __init__(self, viewer, state):
        super().__init__()
        self.viewer = viewer
        self.state = state
        self._frame_width = 0
        self._points_layer = None
        self._updating = False

        layout = QVBoxLayout(self)

        group = QGroupBox("Frame Shift Detection")
        form = QFormLayout(group)

        self.spin_shift_frame = QSpinBox()
        self.spin_shift_frame.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_shift_frame.setMinimum(1)
        self.spin_shift_frame.setMaximum(1)
        self.spin_shift_frame.setValue(1)
        self.spin_shift_frame.setToolTip("Frame index where the shift occurred")
        form.addRow("shift_frame:", self.spin_shift_frame)

        layout.addWidget(group)

        self.btn_load = QPushButton("Load Frames")
        self.btn_load.setToolTip("Shows two consecutive frames side-by-side")
        self.btn_load.clicked.connect(self._load_frames)
        layout.addWidget(self.btn_load)

        instr = QLabel(
            "1. Click a landmark on the LEFT frame\n"
            "2. Click the SAME landmark on the RIGHT frame\n"
            "3. The shift is computed automatically"
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        self.lbl_shift = QLabel("shift_xy = (0, 0)")
        font = self.lbl_shift.font()
        font.setBold(True)
        self.lbl_shift.setFont(font)
        layout.addWidget(self.lbl_shift)

        self.btn_clear = QPushButton("Clear Points")
        self.btn_clear.clicked.connect(self._clear_points)
        layout.addWidget(self.btn_clear)

        layout.addStretch()

    def refresh(self):
        n = len(self.state.all_frames)
        self.spin_shift_frame.setMaximum(max(1, n - 1))

    def _load_frames(self):
        if not self.state.all_frames:
            QMessageBox.warning(self, "No experiment", "Select an experiment first (Tab 0).")
            return

        idx = self.spin_shift_frame.value()
        self.state.shift_frame_idx = idx

        prev_path = os.path.join(self.state.frames_dir, self.state.all_frames[idx - 1])
        curr_path = os.path.join(self.state.frames_dir, self.state.all_frames[idx])
        img_prev = np.array(PILImage.open(prev_path))
        img_curr = np.array(PILImage.open(curr_path))

        self._frame_width = img_prev.shape[1]
        combined = np.concatenate([img_prev, img_curr], axis=1)

        # Remove old layers for this tab
        for name in ["Shift Comparison", "Shift Landmarks"]:
            _remove_layer(self.viewer, name)

        _set_layer(self.viewer, combined, "Shift Comparison",
                   layer_type="image", colormap="gray")

        self._points_layer = self.viewer.add_points(
            np.empty((0, 2)), name="Shift Landmarks",
            size=20, face_color="red", edge_color="white",
        )
        self._points_layer.mode = "add"
        self._points_layer.events.data.connect(self._on_points_changed)

    def _on_points_changed(self, event=None):
        if self._updating or self._points_layer is None:
            return
        pts = self._points_layer.data
        if len(pts) >= 2:
            p1 = pts[-2]  # last two points
            p2 = pts[-1]
            # p1 should be on left half, p2 on right half
            dx = int(round((p2[1] - self._frame_width) - p1[1]))
            dy = int(round(p2[0] - p1[0]))
            self.state.shift_xy = (dx, dy)
            self.lbl_shift.setText(f"shift_xy = ({dx}, {dy})")

    def _clear_points(self):
        if self._points_layer is not None:
            self._updating = True
            self._points_layer.data = np.empty((0, 2))
            self._updating = False
        self.state.shift_xy = (0, 0)
        self.lbl_shift.setText("shift_xy = (0, 0)")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 – Max Distance (Delaunay)
# ═══════════════════════════════════════════════════════════════════════════════

class MaxDistanceTab(QWidget):
    def __init__(self, viewer, state):
        super().__init__()
        self.viewer = viewer
        self.state = state
        self._centroids_by_id = {}

        layout = QVBoxLayout(self)

        group = QGroupBox("Max Distance (Delaunay Neighbours)")
        g_layout = QVBoxLayout(group)

        self.btn_compute = QPushButton("Compute from Segmentation")
        self.btn_compute.setToolTip("Uses the Cellpose preview from Tab 0")
        self.btn_compute.clicked.connect(self._compute)
        g_layout.addWidget(self.btn_compute)

        self.btn_resample = QPushButton("Re-sample Random Cell")
        self.btn_resample.clicked.connect(self._compute)
        g_layout.addWidget(self.btn_resample)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        g_layout.addWidget(self.lbl_info)

        layout.addWidget(group)

        override_group = QGroupBox("Override")
        form = QFormLayout(override_group)

        self.spin_max_dist = QDoubleSpinBox()
        self.spin_max_dist.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_max_dist.setRange(0.1, 9999.0)
        self.spin_max_dist.setDecimals(1)
        self.spin_max_dist.setSingleStep(1.0)
        self.spin_max_dist.setValue(self.state.max_distance)
        self.spin_max_dist.setToolTip("Max pixel distance for trajectory linking")
        self.spin_max_dist.valueChanged.connect(self._on_override)
        form.addRow("max_distance:", self.spin_max_dist)

        layout.addWidget(override_group)
        layout.addStretch()

    def _on_override(self, val):
        self.state.max_distance = val

    def _compute(self):
        seg = self.state.temp_segmentation
        if seg is None:
            QMessageBox.warning(
                self, "No segmentation",
                "Run Cellpose Preview in Tab 0 first to generate a segmentation.")
            return

        cell_ids = np.arange(1, int(seg.max()) + 1)
        raw_centroids = center_of_mass(seg > 0, labels=seg, index=cell_ids)
        self._centroids_by_id = {
            int(cid): rc for cid, rc in zip(cell_ids, raw_centroids)
            if not np.isnan(rc[0])
        }

        if len(self._centroids_by_id) < 4:
            self.lbl_info.setText("Too few cells for Delaunay triangulation.")
            return

        chosen_id = random.choice(list(self._centroids_by_id.keys()))
        cy, cx = self._centroids_by_id[chosen_id]
        neighbor_ids = get_delaunay_neighbors(chosen_id, self._centroids_by_id)

        chosen_rc = np.array([cy, cx])
        nbr_coords = np.array([self._centroids_by_id[n] for n in neighbor_ids])
        dists = np.linalg.norm(nbr_coords - chosen_rc, axis=1)
        mean_dist = float(dists.mean())

        self.state.max_distance = mean_dist
        self.spin_max_dist.setValue(mean_dist)
        self.lbl_info.setText(
            f"Cell {chosen_id}: {len(neighbor_ids)} neighbours\n"
            f"Distances: {np.round(dists, 1).tolist()}\n"
            f"Mean distance: {mean_dist:.1f} px"
        )

        # ── Visualise in napari ───────────────────────────────────────────
        for name in ["Delaunay Highlight", "Delaunay Centroids", "Delaunay Lines"]:
            _remove_layer(self.viewer, name)

        # Highlighted labels
        highlight = np.zeros_like(seg, dtype=np.int32)
        highlight[seg == chosen_id] = 1
        for nid in neighbor_ids:
            highlight[seg == nid] = 2
        _set_layer(self.viewer, highlight, "Delaunay Highlight",
                   layer_type="labels", opacity=0.4)

        # Centroid points
        pts = np.array([[cy, cx]] + [[self._centroids_by_id[n][0],
                                       self._centroids_by_id[n][1]]
                                      for n in neighbor_ids])
        colors = ["cyan"] + ["orange"] * len(neighbor_ids)
        _set_layer(self.viewer, pts, "Delaunay Centroids",
                   layer_type="points", face_color=colors, size=12)

        # Lines to neighbours
        lines = []
        for nid in neighbor_ids:
            nr, nc = self._centroids_by_id[nid]
            lines.append(np.array([[cy, cx], [nr, nc]]))
        if lines:
            _set_layer(self.viewer, lines, "Delaunay Lines",
                       layer_type="shapes", shape_type="line",
                       edge_color="white", edge_width=1, opacity=0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 – Circle ROI
# ═══════════════════════════════════════════════════════════════════════════════

class ROITab(QWidget):
    def __init__(self, viewer, state):
        super().__init__()
        self.viewer = viewer
        self.state = state
        self._updating = False
        self._shapes_layer = None
        self._seg = None
        self._centroid_xs = None
        self._centroid_ys = None
        self._cell_ids = None

        layout = QVBoxLayout(self)

        group = QGroupBox("Circle ROI")
        form = QFormLayout(group)

        self.spin_radius = QSpinBox()
        self.spin_radius.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_radius.setRange(10, 9999)
        self.spin_radius.setValue(self.state.radius)
        self.spin_radius.setToolTip("ROI circle radius in pixels")
        self.spin_radius.valueChanged.connect(self._on_spin_changed)
        form.addRow("radius:", self.spin_radius)

        self.spin_y = QSpinBox()
        self.spin_y.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_y.setRange(-9999, 9999)
        self.spin_y.setValue(self.state.y_shift)
        self.spin_y.setToolTip("Shift centre down (+) or up (-)")
        self.spin_y.valueChanged.connect(self._on_spin_changed)
        form.addRow("y_shift:", self.spin_y)

        self.spin_x = QSpinBox()
        self.spin_x.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_x.setRange(-9999, 9999)
        self.spin_x.setValue(self.state.x_shift)
        self.spin_x.setToolTip("Shift centre right (+) or left (-)")
        self.spin_x.valueChanged.connect(self._on_spin_changed)
        form.addRow("x_shift:", self.spin_x)

        layout.addWidget(group)

        self.btn_show = QPushButton("Show ROI")
        self.btn_show.clicked.connect(self._show_roi)
        layout.addWidget(self.btn_show)

        self.lbl_count = QLabel("")
        font = self.lbl_count.font()
        font.setBold(True)
        self.lbl_count.setFont(font)
        layout.addWidget(self.lbl_count)

        hint = QLabel("Drag the circle in the viewer to reposition.\nUse spinboxes above to resize.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch()

    def _show_roi(self):
        seg = self.state.temp_segmentation
        if seg is None:
            QMessageBox.warning(
                self, "No segmentation",
                "Run Cellpose Preview in Tab 0 first.")
            return

        self._seg = seg
        cell_ids_all = np.unique(seg)
        self._cell_ids = cell_ids_all[cell_ids_all != 0]

        cxs, cys = [], []
        for cid in self._cell_ids:
            ys, xs = np.where(seg == cid)
            cxs.append(xs.mean())
            cys.append(ys.mean())
        self._centroid_xs = np.array(cxs)
        self._centroid_ys = np.array(cys)

        self._update_display()

    def _on_spin_changed(self):
        if self._updating:
            return
        self.state.radius = self.spin_radius.value()
        self.state.y_shift = self.spin_y.value()
        self.state.x_shift = self.spin_x.value()
        if self._seg is not None:
            self._update_display()

    def _update_display(self):
        seg = self._seg
        h, w = seg.shape[:2]
        cx = w / 2 + self.state.x_shift
        cy = h / 2 + self.state.y_shift
        r = self.state.radius

        # Filtered mask
        inside = (self._centroid_xs - cx)**2 + (self._centroid_ys - cy)**2 <= r**2
        valid = self._cell_ids[inside]
        filtered = np.where(np.isin(seg, valid), seg, 0)
        n_inside = len(valid)

        _set_layer(self.viewer, filtered, "ROI Filtered Mask", layer_type="labels", opacity=0.5)
        self.lbl_count.setText(f"{n_inside} cells within ROI")

        # Circle shape (ellipse as bounding box corners: [[r_min, c_min], [r_max, c_max]])
        ellipse = np.array([[cy - r, cx - r], [cy + r, cx + r]])

        for name in ["ROI Circle"]:
            _remove_layer(self.viewer, name)

        self._shapes_layer = self.viewer.add_shapes(
            [ellipse], shape_type="ellipse", name="ROI Circle",
            edge_color="red", edge_width=3, face_color="transparent",
        )
        self._shapes_layer.events.data.connect(self._on_shape_changed)

    def _on_shape_changed(self, event=None):
        if self._updating or self._shapes_layer is None:
            return
        data = self._shapes_layer.data
        if len(data) == 0:
            return

        # napari stores ellipse as 4 corner points
        pts = data[0]
        row_min, col_min = pts.min(axis=0)
        row_max, col_max = pts.max(axis=0)
        cy_new = (row_min + row_max) / 2
        cx_new = (col_min + col_max) / 2
        r_new = int(round((row_max - row_min) / 2))

        seg = self._seg
        h, w = seg.shape[:2]
        y_shift_new = int(round(cy_new - h / 2))
        x_shift_new = int(round(cx_new - w / 2))

        self._updating = True
        self.spin_radius.setValue(r_new)
        self.spin_y.setValue(y_shift_new)
        self.spin_x.setValue(x_shift_new)
        self._updating = False

        self.state.radius = r_new
        self.state.y_shift = y_shift_new
        self.state.x_shift = x_shift_new

        # Update filtered mask
        cx = w / 2 + x_shift_new
        cy = h / 2 + y_shift_new
        inside = (self._centroid_xs - cx)**2 + (self._centroid_ys - cy)**2 <= r_new**2
        valid = self._cell_ids[inside]
        filtered = np.where(np.isin(self._seg, valid), self._seg, 0)
        _set_layer(self.viewer, filtered, "ROI Filtered Mask", layer_type="labels", opacity=0.5)
        self.lbl_count.setText(f"{len(valid)} cells within ROI")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 – Summary + Run Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class SummaryTab(QWidget):
    def __init__(self, viewer, state):
        super().__init__()
        self.viewer = viewer
        self.state = state

        layout = QVBoxLayout(self)

        # ── Summary text ──────────────────────────────────────────────────
        group = QGroupBox("Pipeline Parameters")
        g_layout = QVBoxLayout(group)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumHeight(220)
        font = QFont("Monospace", 9)
        font.setStyleHint(QFont.Monospace)
        self.txt_summary.setFont(font)
        g_layout.addWidget(self.txt_summary)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Summary")
        self.btn_refresh.clicked.connect(self.refresh)
        btn_row.addWidget(self.btn_refresh)

        self.btn_copy = QPushButton("Copy to Clipboard")
        self.btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_row.addWidget(self.btn_copy)
        g_layout.addLayout(btn_row)

        layout.addWidget(group)

        # ── Grace period ──────────────────────────────────────────────────
        extra_group = QGroupBox("Additional Parameters")
        extra_form = QFormLayout(extra_group)

        self.spin_grace = QSpinBox()
        self.spin_grace.setMaximumWidth(SPIN_MAX_WIDTH)
        self.spin_grace.setRange(1, 100)
        self.spin_grace.setValue(self.state.grace_period)
        self.spin_grace.setToolTip("Number of frames to look back when linking trajectories")
        self.spin_grace.valueChanged.connect(lambda v: setattr(self.state, 'grace_period', v))
        extra_form.addRow("grace_period:", self.spin_grace)

        layout.addWidget(extra_group)

        # ── Export / Run ──────────────────────────────────────────────────
        run_group = QGroupBox("Export & Run")
        run_layout = QVBoxLayout(run_group)

        self.btn_export = QPushButton("Export Shell Script")
        self.btn_export.setToolTip("Save a customised run_processes.sh")
        self.btn_export.clicked.connect(self._export_script)
        run_layout.addWidget(self.btn_export)

        self.chk_seg = QCheckBox("Run Segmentation")
        self.chk_seg.setChecked(True)
        run_layout.addWidget(self.chk_seg)

        self.chk_traj = QCheckBox("Run Trajectories")
        self.chk_traj.setChecked(True)
        run_layout.addWidget(self.chk_traj)

        self.chk_pre = QCheckBox("Run Pre-Analysis")
        self.chk_pre.setChecked(True)
        run_layout.addWidget(self.chk_pre)

        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.clicked.connect(self._run_pipeline)
        run_layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        run_layout.addWidget(self.progress)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(150)
        font_log = QFont("Monospace", 8)
        font_log.setStyleHint(QFont.Monospace)
        self.txt_log.setFont(font_log)
        run_layout.addWidget(self.txt_log)

        layout.addWidget(run_group)
        layout.addStretch()

    def _build_summary(self):
        s = self.state
        lines = [
            "=" * 42,
            "  Pipeline Parameters",
            "=" * 42,
            f"  GLOBAL_DIR           = {s.global_dir}",
            f"  flow_threshold       = {s.flow_threshold}",
            f"  cellprob_threshold   = {s.cellprob_threshold}",
            f"  niter                = {s.niter}",
            f"  diameter             = {s.diameter}",
            f"  save_interval        = {s.save_interval}",
            f"  shift_frame          = {s.shift_frame_idx}",
            f"  shift_xy             = {s.shift_xy[0]} {s.shift_xy[1]}",
            f"  max_distance         = {s.max_distance:.1f}",
            f"  radius               = {s.radius}",
            f"  y_shift (radius_y)   = {s.y_shift}",
            f"  x_shift (radius_x)   = {s.x_shift}",
            f"  grace_period         = {s.grace_period}",
            "=" * 42,
        ]
        return "\n".join(lines)

    def refresh(self):
        self.txt_summary.setPlainText(self._build_summary())

    def _copy_to_clipboard(self):
        self.refresh()
        QApplication.clipboard().setText(self.txt_summary.toPlainText())

    def _generate_script(self):
        s = self.state
        rel_dir = os.path.relpath(s.global_dir, os.path.dirname(__file__))
        return f"""#!/bin/bash
set -euo pipefail

# -----------------------------
# PATHS
# -----------------------------
GLOBAL_DIR="{rel_dir}"
IMAGE_DIR="${{GLOBAL_DIR}}/frames"
MASK_DIR="${{GLOBAL_DIR}}/masks"
SAVE_PATH="${{GLOBAL_DIR}}/analysis"

SCRIPT1="SCRIPTS/core_pipeline/segmentation.py"
SCRIPT2="SCRIPTS/core_pipeline/trajectories.py"

# -----------------------------
# CELLPOSE PARAMETERS
# -----------------------------
FLOW_THRESHOLD={s.flow_threshold}
CELLPROB_THRESHOLD={s.cellprob_threshold}
NITER={s.niter}
DIAMETER={s.diameter}

# -----------------------------
# TRAJECTORY PARAMETERS
# -----------------------------
MAX_DISTANCE={s.max_distance:.1f}
GRACE_PERIOD={s.grace_period}
RADIUS={s.radius}
RADIUS_Y={s.y_shift}
RADIUS_X={s.x_shift}
SHIFT_FRAME={s.shift_frame_idx}
SHIFT_XY="{s.shift_xy[0]} {s.shift_xy[1]}"
SAVE_INTERVAL={s.save_interval}

# -----------------------------
# Ensure scripts exist
# -----------------------------
if [[ ! -f "$SCRIPT1" || ! -f "$SCRIPT2" ]]; then
    echo "One or more scripts not found."
    exit 1
fi

# -----------------------------
# Log config
# -----------------------------
mkdir -p "$SAVE_PATH"
CONFIG_FILE="${{SAVE_PATH}}/config.txt"
cat > "$CONFIG_FILE" <<CFGEOF
Run date: $(date)

[PATHS]
GLOBAL_DIR=$GLOBAL_DIR
IMAGE_DIR=$IMAGE_DIR
MASK_DIR=$MASK_DIR
SAVE_PATH=$SAVE_PATH

[CELLPOSE]
FLOW_THRESHOLD=$FLOW_THRESHOLD
CELLPROB_THRESHOLD=$CELLPROB_THRESHOLD
NITER=$NITER
DIAMETER=$DIAMETER

[TRAJECTORIES]
MAX_DISTANCE=$MAX_DISTANCE
GRACE_PERIOD=$GRACE_PERIOD
RADIUS=$RADIUS
RADIUS_Y=$RADIUS_Y
RADIUS_X=$RADIUS_X
SHIFT_FRAME=$SHIFT_FRAME
SHIFT_XY=$SHIFT_XY
SAVE_INTERVAL=$SAVE_INTERVAL
CFGEOF
echo "Config saved to $CONFIG_FILE"

echo "--- Accessing ${{GLOBAL_DIR}} ---"

# -----------------------------
# Run segmentation
# -----------------------------
echo "Starting cellpose segmentation..."
python3 "$SCRIPT1" \\
    --image_dir "$IMAGE_DIR" \\
    --mask_dir "$MASK_DIR" \\
    --flow_threshold "$FLOW_THRESHOLD" \\
    --cellprob_threshold "$CELLPROB_THRESHOLD" \\
    --niter "$NITER" \\
    --diameter "$DIAMETER" &
PID1=$!

sleep 5

# -----------------------------
# Run trajectory processing
# -----------------------------
echo "Starting trajectory processing..."
python3 "$SCRIPT2" \\
    --mask_dir "$MASK_DIR" \\
    --image_dir "$IMAGE_DIR" \\
    --save_path "$SAVE_PATH" \\
    --max_distance "$MAX_DISTANCE" \\
    --grace_period "$GRACE_PERIOD" \\
    --radius "$RADIUS" \\
    --radius_y "$RADIUS_Y" \\
    --radius_x "$RADIUS_X" \\
    --shift_frame "$SHIFT_FRAME" \\
    --shift_xy $SHIFT_XY \\
    --save_interval "$SAVE_INTERVAL" &
PID2=$!

wait $PID1 $PID2

# -----------------------------
# Pre-analysis plots
# -----------------------------
echo ">>> Running pre-analysis plots for ${{GLOBAL_DIR}}"
python3 SCRIPTS/core_pipeline/PreAnalysis.py \\
    --exp "$GLOBAL_DIR" \\
    --analysis_dir "$SAVE_PATH"
"""

    def _export_script(self):
        if not self.state.global_dir:
            QMessageBox.warning(self, "No experiment", "Select an experiment first (Tab 0).")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Shell Script", "run_processes.sh", "Shell Scripts (*.sh)")
        if not path:
            return
        with open(path, "w") as f:
            f.write(self._generate_script())
        os.chmod(path, 0o755)
        self.txt_log.append(f"Exported: {path}")

    def _run_pipeline(self):
        if not self.state.global_dir:
            QMessageBox.warning(self, "No experiment", "Select an experiment first (Tab 0).")
            return

        s = self.state
        project_root = os.path.dirname(os.path.abspath(__file__))
        cmds = []

        if self.chk_seg.isChecked():
            cmds.append((
                "Segmentation",
                [sys.executable, "SCRIPTS/core_pipeline/segmentation.py",
                 "--image_dir", s.frames_dir,
                 "--mask_dir", s.masks_dir,
                 "--flow_threshold", str(s.flow_threshold),
                 "--cellprob_threshold", str(s.cellprob_threshold),
                 "--niter", str(s.niter),
                 "--diameter", str(s.diameter)],
            ))

        if self.chk_traj.isChecked():
            cmds.append((
                "Trajectories",
                [sys.executable, "SCRIPTS/core_pipeline/trajectories.py",
                 "--mask_dir", s.masks_dir,
                 "--image_dir", s.frames_dir,
                 "--save_path", s.save_path,
                 "--max_distance", str(s.max_distance),
                 "--grace_period", str(s.grace_period),
                 "--radius", str(s.radius),
                 "--radius_y", str(s.y_shift),
                 "--radius_x", str(s.x_shift),
                 "--shift_frame", str(s.shift_frame_idx),
                 "--shift_xy", str(s.shift_xy[0]), str(s.shift_xy[1]),
                 "--save_interval", str(s.save_interval)],
            ))

        if self.chk_pre.isChecked():
            cmds.append((
                "Pre-Analysis",
                [sys.executable, "SCRIPTS/core_pipeline/PreAnalysis.py",
                 "--exp", s.global_dir,
                 "--analysis_dir", s.save_path],
            ))

        if not cmds:
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.txt_log.clear()

        @thread_worker
        def _work():
            results = []
            for label, cmd in cmds:
                results.append(("start", label))
                try:
                    proc = subprocess.run(
                        cmd, cwd=project_root,
                        capture_output=True, text=True, timeout=7200,
                    )
                    results.append(("stdout", proc.stdout))
                    if proc.stderr:
                        results.append(("stderr", proc.stderr))
                    results.append(("done", label, proc.returncode))
                except Exception as e:
                    results.append(("error", label, str(e)))
            return results

        worker = _work()
        worker.returned.connect(self._on_pipeline_done)
        worker.start()

    def _on_pipeline_done(self, results):
        for item in results:
            if item[0] == "start":
                self.txt_log.append(f"\n--- {item[1]} ---")
            elif item[0] == "stdout":
                if item[1].strip():
                    self.txt_log.append(item[1].strip())
            elif item[0] == "stderr":
                self.txt_log.append(f"[stderr] {item[1].strip()}")
            elif item[0] == "done":
                rc = item[2]
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                self.txt_log.append(f"{item[1]}: {status}")
            elif item[0] == "error":
                self.txt_log.append(f"{item[1]}: ERROR - {item[2]}")

        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 6 – Duplicate Centers / Masks
# ═══════════════════════════════════════════════════════════════════════════════

class DuplicateTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state

        layout = QVBoxLayout(self)

        group = QGroupBox("Duplicate Centers / Masks")
        form = QFormLayout(group)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["centers", "masks", "both"])
        form.addRow("Mode:", self.combo_mode)

        self.combo_source = QComboBox()
        self.combo_source.addItems(["temp (from Cellpose preview)", "masks_dir (experiment)"])
        form.addRow("Mask source:", self.combo_source)

        layout.addWidget(group)

        self.btn_run = QPushButton("Run Duplication")
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    def _run(self):
        s = self.state
        if not s.all_frames:
            QMessageBox.warning(self, "No experiment", "Select an experiment first (Tab 0).")
            return

        mode = self.combo_mode.currentText()
        use_temp = self.combo_source.currentIndex() == 0

        frames = s.all_frames
        masks_dir = s.masks_dir
        analysis_dir = s.save_path

        # ── Load mask ────────────────────────────────────────────────────
        mask = None
        if mode in ("masks", "both"):
            if use_temp:
                if s.temp_segmentation is None:
                    QMessageBox.warning(self, "No temp mask",
                                        "Run Cellpose Preview in Tab 0 first.")
                    return
                mask = s.temp_segmentation
            else:
                if not os.path.isdir(masks_dir):
                    QMessageBox.warning(self, "No masks dir",
                                        f"Masks directory not found:\n{masks_dir}")
                    return
                mask_files = sorted(os.listdir(masks_dir))
                if not mask_files:
                    QMessageBox.warning(self, "No masks", "No mask files found.")
                    return
                mask = load_segmentation(os.path.join(masks_dir, mask_files[0]))

        # ── Load centers ─────────────────────────────────────────────────
        centers = None
        first_centers_file = None
        centers_dir = None
        if mode in ("centers", "both"):
            centers_dir = os.path.join(analysis_dir, "cellpose_centers")
            if not os.path.isdir(centers_dir):
                QMessageBox.warning(self, "No centers dir",
                                    f"Centers directory not found:\n{centers_dir}")
                return
            c_files = sorted(os.listdir(centers_dir))
            if not c_files:
                QMessageBox.warning(self, "No centers", "No center files found.")
                return
            first_centers_file = c_files[0]
            centers = np.load(os.path.join(centers_dir, first_centers_file), allow_pickle=True)

        # ── Duplicate ────────────────────────────────────────────────────
        self.btn_run.setEnabled(False)
        self.progress.setRange(0, len(frames))
        self.progress.setValue(0)
        self.progress.setVisible(True)

        os.makedirs(masks_dir, exist_ok=True)
        if centers_dir:
            os.makedirs(centers_dir, exist_ok=True)

        @thread_worker
        def _work():
            for i, fname in enumerate(frames):
                base = os.path.splitext(fname)[0]
                if mode in ("centers", "both") and centers is not None:
                    np.save(os.path.join(centers_dir, f"{base}_centers.npy"), centers)
                if mode in ("masks", "both") and mask is not None:
                    np.save(os.path.join(masks_dir, f"{base}.npy"), mask)
            return len(frames)

        worker = _work()
        worker.returned.connect(self._on_done)
        worker.yielded.connect(lambda i: self.progress.setValue(i))
        worker.start()

    def _on_done(self, count):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.lbl_status.setText(f"Done! Duplicated to {count} frames.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main – wire everything together
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    viewer = napari.Viewer(title="Cell Analysis – Preprocessing")
    state = PipelineState()

    tabs = QTabWidget()

    # Callbacks for cross-tab refresh
    def on_experiment_loaded():
        tab_interval.refresh()
        tab_shift.refresh()
        tab_summary.refresh()

    tab_cellpose = CellposeTab(viewer, state, on_experiment_loaded=on_experiment_loaded)
    tab_interval = SaveIntervalTab(state)
    tab_shift = ShiftTab(viewer, state)
    tab_maxdist = MaxDistanceTab(viewer, state)
    tab_roi = ROITab(viewer, state)
    tab_summary = SummaryTab(viewer, state)
    tab_dup = DuplicateTab(state)

    tabs.addTab(tab_cellpose, "0. Cellpose")
    tabs.addTab(tab_interval, "1. Save Interval")
    tabs.addTab(tab_shift, "2. Frame Shift")
    tabs.addTab(tab_maxdist, "3. Max Distance")
    tabs.addTab(tab_roi, "4. Circle ROI")
    tabs.addTab(tab_summary, "5. Summary / Run")
    tabs.addTab(tab_dup, "6. Duplicate")

    # Refresh summary when switching to it
    tabs.currentChanged.connect(
        lambda idx: tab_summary.refresh() if idx == 5 else None
    )

    viewer.window.add_dock_widget(tabs, name="Preprocessing", area="right")

    napari.run()


if __name__ == "__main__":
    main()
