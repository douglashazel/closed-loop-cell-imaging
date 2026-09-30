"""
Dry-run frame source (config "dry_run": true). run_system.sh starts this in
place of the microscope: every FRAME_INTERVAL_SEC it writes one synthetic PNG
per channel into watch_dir, named like the scope's files
(channel_{c}_image_0_a_timepoint_{t:05d}.png).

Cell brightness cycles between 1x and 3x its frame-0 level, so it crosses the
setpoint CreateDecisions derives (2x frame 0) and triggers acid pulses; the
two channels cycle at different rates so every NN/AN/NA/AA state occurs.
At startup it also writes a matching frame-0 label mask into mask_dir, so
Push mask in the web GUI works without running Cellpose.
"""
import os
import time
import numpy as np
from PIL import Image

from io_utils import log, load_config, parse_filename

cfg = load_config()
watch_dir    = cfg["watch_dir"]
mask_dir     = cfg["mask_dir"]
num_channels = cfg["num_channels"]

FRAME_INTERVAL_SEC = 5.0
SIZE = 256

if not cfg.get("dry_run", False):
    log("fake_frames: dry_run is false in config.json -- refusing to write synthetic frames")
    raise SystemExit(1)


def cell_labels(seed):
    """Label image with a dozen round 'cells' (0 = background)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    labels = np.zeros((SIZE, SIZE), dtype=np.int32)
    for cell_id in range(1, 13):
        cy, cx = rng.integers(20, SIZE - 20, size=2)
        r = rng.integers(8, 16)
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = cell_id
    return labels


def frame_image(labels, t, cycle_frames, rng):
    """8-bit frame: noisy background plus cells at 1x..3x frame-0 brightness."""
    gain = 2.0 - np.cos(2 * np.pi * t / cycle_frames)
    img = rng.normal(10, 2, size=labels.shape)
    img[labels > 0] += 40 * gain
    return np.clip(img, 0, 255).astype(np.uint8)


os.makedirs(watch_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

labels = {ch: cell_labels(ch) for ch in range(1, num_channels + 1)}
for ch, lab in labels.items():
    mask_path = os.path.join(mask_dir, f"00000_channel{ch}.npy")
    if not os.path.exists(mask_path):
        np.save(mask_path, lab)

# Continue numbering after any frames an earlier dry run left behind
existing = [parse_filename(f)[1] for f in os.listdir(watch_dir)]
existing = [fr for fr in existing if fr is not None]
t = max(existing) + 1 if existing else 0
rng = np.random.default_rng()
log(f"fake_frames: writing synthetic frames to {watch_dir} every "
    f"{FRAME_INTERVAL_SEC:.0f} s, starting at timepoint {t}")

while True:
    for ch, lab in labels.items():
        name = f"channel_{ch}_image_0_a_timepoint_{t:05d}.png"
        tmp = os.path.join(watch_dir, name + ".tmp")
        Image.fromarray(frame_image(lab, t, 24 + 12 * (ch - 1), rng)).save(tmp, format="PNG")
        os.replace(tmp, os.path.join(watch_dir, name))  # readers never see a partial PNG
    t += 1
    time.sleep(FRAME_INTERVAL_SEC)
