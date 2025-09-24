import os
import re
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from cellpose import models, io, utils
from scipy.ndimage import binary_dilation

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# --- Load Config ---
with open("config.json", "r") as f:
    cfg = json.load(f)

watch_dir = cfg["watch_dir"]
mask_dir = cfg["mask_dir"]
temp_overlays = cfg["temp_overlays"]

log("Cellpose model loading complete.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Track processed images by checking existing .npy files
existing_masks = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith('.npy')}
processed = {os.path.join(watch_dir, f) for f in os.listdir(watch_dir) if os.path.splitext(f)[0] in existing_masks}

model = models.CellposeModel(gpu=True)

def parse_filename(fname):
    """Extract frame and channel from filename like channel_1_image_0_a_timepoint_00000.png"""
    base = os.path.splitext(fname)[0]
    m = re.search(r'channel_(\d+).*timepoint_(\d+)$', base)
    if not m:
        raise ValueError(f"Unexpected filename format: {fname}")
    channel = int(m.group(1))
    frame = int(m.group(2))
    return frame, channel, base

while True:
    images = sorted([f for f in os.listdir(watch_dir) if f.endswith(('.png', '.jpg'))])
    for f in images:
        path = os.path.join(watch_dir, f)
        if path not in processed:
            log(f'Processing {f}...')
            start_time = time.time()

            retries = 0
            while retries < cfg["num_tries"]:
                try:
                    # Parse filename
                    frame, channel, base_name = parse_filename(f)

                    # Load image
                    img = io.imread(path)

                    # Run Cellpose segmentation
                    masks, _, _ = model.eval([img])
                    masks = masks[0]
                    
                    # Save segmentation with parsed naming
                    save_base = f"{frame:05d}_channel{channel}"
                    save_path = os.path.join(mask_dir, save_base + '.npy')
                    np.save(save_path, masks)

                    # also update overlay for initial frame
                    if frame == 0:
                        outlines = utils.masks_to_outlines(masks)
                        outlines = binary_dilation(outlines, iterations=3)
                        overlay = img.copy()
                        if overlay.ndim == 2:
                            overlay = np.stack([overlay] * 3, axis=-1)
                        overlay[outlines] = [255, 0, 0]

                        fig = plt.figure(dpi=300)
                        plt.title(save_base)
                        plt.imshow(overlay)
                        plt.axis("off")
                        overlay_path = os.path.join(temp_overlays, f"{save_base}_overlay.png")
                        fig.savefig(overlay_path, dpi=300, bbox_inches="tight")

                    processed.add(path)

                    elapsed = time.time() - start_time
                    log(f'Done {f} in {elapsed:.2f} seconds')
                    break  # success

                except (OSError, ValueError, EOFError, AttributeError) as e:
                    retries += 1
                    log(f"Retry {retries}/{cfg['num_tries']} for {f}: {e}")
                    time.sleep(cfg["sleep_time"])

            else:
                log(f"Failed to process {f} after {cfg['num_tries']} retries. Skipping.")