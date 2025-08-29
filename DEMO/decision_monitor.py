import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

num_channels = 6
watch_dir = 'incoming_frames'
mask_dir = 'processed_masks'
decision_dir = 'temp_decisions'
final_dir = 'final_decisions'
os.makedirs(decision_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Wait for all channels' images for a given frame
def wait_for_images(frame):
    frame_str = f"{frame:03d}"
    while True:
        imgs = [f for f in os.listdir(watch_dir) if f.startswith(frame_str) and f.endswith('.png')]
        if len(imgs) >= num_channels:
            return
        time.sleep(2)

# Compute dynamic setpoint using frame 000
def compute_setpoint(initial_masks):
    setpoint_vals = []
    for ch in range(1, num_channels+1):
        img_path = os.path.join(watch_dir, f"000_channel{ch}.png")
        while not os.path.exists(img_path):
            time.sleep(2)
        img = np.array(Image.open(img_path), dtype=np.float32)
        mask = initial_masks[ch]
        setpoint_vals.append(img[mask].mean())
    avg_setpoint = np.mean(setpoint_vals)
    return avg_setpoint

# Find last processed frame
def last_processed_frame():
    frames = []
    for f in os.listdir(final_dir):
        if f.endswith('.csv'):
            frame_idx = f.split('.')[0]
            if frame_idx.isdigit():
                frames.append(int(frame_idx))
    return max(frames) if frames else -1

# Process a frame using preloaded masks
def process_frame(frame, acidic_media, basic_media, initial_masks):
    frame_str = f"{frame:03d}"
    wait_for_images(frame)  # Only wait for new images, not segmentation

    for ch in range(1, num_channels+1):
        img_path = os.path.join(watch_dir, f"{frame_str}_channel{ch}.png")
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path), dtype=np.float32)
        mask = initial_masks[ch]  # Use the segmentation from frame 000
        mean_val = img[mask].mean()

        if basic_media < mean_val < acidic_media:
            decision = 'no change'
        elif mean_val <= basic_media:
            decision = 'add basic media'
        else:
            decision = 'add acidic media'

        with open(os.path.join(decision_dir, f"{frame_str}_channel{ch}.txt"), 'w') as f:
            f.write(decision)

        log(f"{frame_str}_channel{ch}: {mean_val:.3f} -> {decision}")

# Combine decisions into final CSV
def finalize_decisions(frame):
    frame_str = f"{frame:03d}"
    while True:
        decs = [f for f in os.listdir(decision_dir) if f.startswith(frame_str) and f.endswith('.txt')]
        if len(decs) >= num_channels:
            break
        time.sleep(2)

    decisions = []
    for ch in range(1, num_channels+1):
        dec_path = os.path.join(decision_dir, f"{frame_str}_channel{ch}.txt")
        with open(dec_path, 'r') as f:
            decisions.append(f.read().strip())
    df = pd.DataFrame({'channel': list(range(num_channels)), 'decision': decisions})
    df.to_csv(os.path.join(final_dir, f"{frame_str}.csv"), index=False)
    log(f"Finalized decisions for frame {frame_str}")

# ---- INITIAL SETUP ----
log("Waiting for initial frame (000) masks and images to compute setpoint...")

# Wait for all masks for frame 000
while True:
    masks = [f for f in os.listdir(mask_dir) if f.startswith("000") and f.endswith('.npy')]
    if len(masks) >= num_channels:
        break
    time.sleep(2)

# Load initial masks into memory
initial_masks = {}
for ch in range(1, num_channels+1):
    mask_path = os.path.join(mask_dir, f"000_channel{ch}.npy")
    while not os.path.exists(mask_path):
        time.sleep(2)
    initial_masks[ch] = np.load(mask_path) > 0

# Compute setpoint using frame 000
setpoint = compute_setpoint(initial_masks)
acidic_media = setpoint * 1.05
basic_media = setpoint * 0.95
log(f"Setpoint computed: {setpoint:.3f}, basic={basic_media:.3f}, acidic={acidic_media:.3f}")

# ---- MAIN LOOP ----
while True:
    last_frame = last_processed_frame()
    # Look for frames based on image presence (not mask presence anymore)
    frame_indices = sorted(set(int(f.split('_channel')[0]) for f in os.listdir(watch_dir) if f.endswith('.png')))
    next_frames = [f for f in frame_indices if f > last_frame]

    if not next_frames:
        time.sleep(5)
        continue

    for frame in next_frames:
        frame_str = f"{frame:03d}"
        lock_path = os.path.join(final_dir, f"{frame_str}.lock")

        if os.path.exists(lock_path):
            continue

        open(lock_path, 'w').close()  # Create lock

        process_frame(frame, acidic_media, basic_media, initial_masks)
        finalize_decisions(frame)

        os.remove(lock_path)