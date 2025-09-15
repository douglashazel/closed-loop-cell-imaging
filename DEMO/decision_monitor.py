import os
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

with open("config.json", "r") as f:
    cfg = json.load(f)

num_channels = cfg["num_channels"]
watch_dir = cfg["watch_dir"]
mask_dir = cfg["mask_dir"]
curr_mask_dir = cfg["curr_mask_dir"]
decision_dir = cfg["decision_dir"]
final_dir = cfg["final_dir"]
setpoint_file = cfg["setpoint_file"]

decision_key = cfg['decision_key']
decision_rev = {v: k for k, v in decision_key.items()}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def wait_for_images(frame):
    frame_str = f"{frame:03d}"
    while True:
        imgs = [f for f in os.listdir(watch_dir) if f.startswith(frame_str) and f.endswith('.png')]
        if len(imgs) >= num_channels:
            return
        time.sleep(2)

def compute_setpoint(initial_masks):
    setpoint_vals = []
    for ch in range(1, num_channels+1):
        img_path = os.path.join(watch_dir, f"000_channel{ch}.png")
        while not os.path.exists(img_path):
            time.sleep(2)
        img = np.array(Image.open(img_path), dtype=np.float32)
        mask = initial_masks[ch]
        setpoint_vals.append(img[mask].mean())
    avg_setpoint = np.mean(setpoint_vals) * 0.8
    return avg_setpoint

def last_processed_frame():
    frames = []
    for f in os.listdir(final_dir):
        if f.endswith('.csv'):
            frame_idx = f.split('.')[0]
            if frame_idx.isdigit():
                frames.append(int(frame_idx))
    return max(frames) if frames else -1

def save_setpoints(setpoint, basic, acidic):
    with open(setpoint_file, 'w') as f:
        f.write(f"setpoint={setpoint:.6f}\n")
        f.write(f"basic={basic:.6f}\n")
        f.write(f"acidic={acidic:.6f}\n")

def load_setpoints(default_setpoint, default_basic, default_acidic):
    if not os.path.exists(setpoint_file):
        return default_setpoint, default_basic, default_acidic
    vals = {'setpoint': default_setpoint,
            'basic': default_basic,
            'acidic': default_acidic}
    try:
        with open(setpoint_file, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=')
                    if k in vals:
                        vals[k] = float(v)
    except Exception as e:
        log(f"Warning: failed to parse {setpoint_file}: {e}")
    return vals['setpoint'], vals['basic'], vals['acidic']

def process_frame(frame, default_setpoint, default_basic, default_acidic):
    frame_str = f"{frame:03d}"
    wait_for_images(frame)

    setpoint, basic_media, acidic_media = load_setpoints(default_setpoint, default_basic, default_acidic)

    for ch in range(1, num_channels+1):
        img_path = os.path.join(watch_dir, f"{frame_str}_channel{ch}.png")
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path), dtype=np.float32)

        # load the *current* mask for this channel (whatever frame it came from)
        mask_file = [f for f in os.listdir(curr_mask_dir) if f.endswith(f"_channel{ch}.npy")]
        if not mask_file:
            log(f"No mask found for channel {ch} in {curr_mask_dir}, skipping")
            continue
        mask_path = os.path.join(curr_mask_dir, mask_file[0])
        mask = np.load(mask_path) > 0

        mean_val = img[mask].mean()

        if basic_media < mean_val < acidic_media:
            decision = decision_key['add neutral media']
        elif mean_val <= basic_media:
            decision = decision_key['add basic media']
        else:
            decision = decision_key['add acidic media']

        with open(os.path.join(decision_dir, f"{frame_str}_channel{ch}.txt"), 'w') as f:
            f.write(str(decision))

        log(f"{frame_str}_channel{ch}: {mean_val:.3f} -> {decision_rev[decision]} "
            f"(setpoint={setpoint:.3f}, basic={basic_media:.3f}, acidic={acidic_media:.3f})")

def finalize_decisions(frame):
    frame_str = f"{frame:03d}"
    while True:
        decs = [f for f in os.listdir(decision_dir) if f.startswith(frame_str) and f.endswith('.txt')]
        if len(decs) >= num_channels:
            break
        time.sleep(2)

    channels = []
    decisions = []
    for ch in range(1, num_channels+1):
        dec_path = os.path.join(decision_dir, f"{frame_str}_channel{ch}.txt")
        with open(dec_path, 'r') as f:
            decision_val = int(f.read().strip())
        channels.append(ch)  # 1..6
        decisions.append(decision_val)  # keep as int

    df = pd.DataFrame({
        "channel": channels,
        "decision": decisions
    })
    df.to_csv(os.path.join(final_dir, f"{frame_str}.csv"), index=False)
    log(f"Finalized decisions for frame {frame_str}")

# ---- INITIAL SETUP ----
log("Waiting for initial frame (000) masks and images to compute setpoint...")

while True:
    masks = [f for f in os.listdir(mask_dir) if f.startswith("000") and f.endswith('.npy')]
    if len(masks) >= num_channels:
        break
    time.sleep(2)

initial_masks = {}
for ch in range(1, num_channels+1):
    mask_path = os.path.join(curr_mask_dir, f"000_channel{ch}.npy")
    while not os.path.exists(mask_path):
        time.sleep(2)
    initial_masks[ch] = np.load(mask_path) > 0

setpoint = compute_setpoint(initial_masks)
acidic_media = setpoint * 1.05
basic_media = setpoint * 0.95

save_setpoints(setpoint, basic_media, acidic_media)
log(f"Setpoint computed: {setpoint:.3f}, basic={basic_media:.3f}, acidic={acidic_media:.3f} "
    f"and saved to {setpoint_file}")

# ---- MAIN LOOP ----
while True:
    last_frame = last_processed_frame()
    frame_indices = sorted(set(int(f.split('_channel')[0]) for f in os.listdir(watch_dir) if f.endswith('.png')))
    next_frames = [f for f in frame_indices if f > last_frame]

    if not next_frames:
        time.sleep(3)
        continue

    for frame in next_frames:
        frame_str = f"{frame:03d}"
        lock_path = os.path.join(final_dir, f"{frame_str}.lock")
        if os.path.exists(lock_path):
            continue

        open(lock_path, 'w').close()
        process_frame(frame, setpoint, basic_media, acidic_media)
        finalize_decisions(frame)
        os.remove(lock_path)