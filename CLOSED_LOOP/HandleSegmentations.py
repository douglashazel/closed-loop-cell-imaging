import os
import time
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io, utils
from scipy.ndimage import binary_dilation

from io_utils import log, load_config, parse_filename

cfg = load_config()
watch_dir            = cfg["watch_dir"]
mask_dir             = cfg["mask_dir"]
temp_overlays        = cfg["temp_overlays"]
continuous_seg       = cfg.get("continuous_segmentation", True)

# curr_mask_dir is written exclusively by preprocess.ipynb (the user "push").
# This script only archives masks to mask_dir.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log("Loading Cellpose model...")
model = models.CellposeModel(gpu=True)
log("Cellpose model ready.")
if not continuous_seg:
    log("continuous_segmentation=False — will exit after frame 0 is fully segmented.")

def mask_exists(frame, channel):
    return os.path.exists(os.path.join(mask_dir, f"{frame:05d}_channel{channel}.npy"))

def save_overlay(img, masks, save_base):
    outlines = binary_dilation(utils.masks_to_outlines(masks), iterations=3)
    overlay = img.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    overlay[outlines] = [255, 0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].imshow(overlay);         axes[0].set_title("With segmentation"); axes[0].axis("off")
    axes[1].imshow(img, cmap="gray"); axes[1].set_title("Raw image");         axes[1].axis("off")
    fig.savefig(os.path.join(temp_overlays, f"{save_base}_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

while True:
    images = sorted(f for f in os.listdir(watch_dir) if f.lower().endswith(('.png', '.jpg')))
    new_work = False

    for fname in images:
        channel, frame = parse_filename(fname)
        if channel is None or mask_exists(frame, channel):
            continue

        # When continuous_segmentation is False, only process frame 0
        if not continuous_seg and frame > 0:
            continue

        new_work = True
        path = os.path.join(watch_dir, fname)
        log(f"Processing {fname}...")
        start = time.time()

        for attempt in range(cfg["num_tries"]):
            try:
                img = io.imread(path)
                masks, _, _ = model.eval([img])
                masks = masks[0]

                save_base = f"{frame:05d}_channel{channel}"

                # Save to archive only — curr_mask_dir is owned by preprocess.ipynb
                np.save(os.path.join(mask_dir, save_base + ".npy"), masks)
                
                # Write ROI metadata for MonitorPerformance.py
                import json
                num_cells = len(np.unique(masks)) - 1
                with open(os.path.join(mask_dir, save_base + "_meta.json"), "w") as f:
                    json.dump({"roi_count": int(num_cells)}, f)

                if frame == 0:
                    save_overlay(img, masks, save_base)

                log(f"Done {fname} in {time.time() - start:.2f}s")
                break

            except (OSError, ValueError, EOFError, AttributeError) as e:
                log(f"Retry {attempt + 1}/{cfg['num_tries']} for {fname}: {e}")
                time.sleep(cfg["sleep_time"])
        else:
            log(f"Failed {fname} after {cfg['num_tries']} retries. Skipping.")

    # In non-continuous mode, exit once all frame-0 channels are segmented
    if not continuous_seg:
        num_channels = cfg["num_channels"]
        done = all(mask_exists(0, ch) for ch in range(1, num_channels + 1))
        if done:
            log("Frame 0 fully segmented. continuous_segmentation=False — exiting.")
            break

    if not new_work:
        time.sleep(cfg["sleep_time"])
