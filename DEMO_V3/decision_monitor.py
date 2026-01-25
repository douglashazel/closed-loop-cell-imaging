import os
import re
import time
import json
import tomlkit
import numpy as np
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

# regex for filenames like channel_1_image_0_a_timepoint_00000.png
FILENAME_PATTERN = re.compile(r"channel_(\d+).*timepoint_(\d+)\.png")

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def parse_filename(fname):
    m = FILENAME_PATTERN.match(fname)
    if not m:
        return None, None
    channel = int(m.group(1))
    frame = int(m.group(2))
    return channel, frame

def wait_for_images(frame):
    while True:
        imgs = []
        for f in os.listdir(watch_dir):
            if f.endswith('.png'):
                ch, fr = parse_filename(f)
                if fr == frame:
                    imgs.append(f)
        if len(imgs) >= num_channels:
            return
        time.sleep(cfg["sleep_time"])

def compute_setpoint(initial_masks):
    setpoint_vals = []
    for ch in range(1, num_channels+1):
        # find image for frame 0 and this channel
        img_file = next(
            (f for f in os.listdir(watch_dir) if f.endswith('.png')
             and parse_filename(f) == (ch, 0)), None
        )
        if img_file is None:
            continue
        img_path = os.path.join(watch_dir, img_file)
        while not os.path.exists(img_path):
            time.sleep(cfg["sleep_time"])
        img = np.array(Image.open(img_path), dtype=np.float32)
        mask = initial_masks[ch]
        setpoint_vals.append(img[mask].mean())
    avg_setpoint = np.mean(setpoint_vals) * 0.8
    return avg_setpoint

def last_processed_frame():
    frames = []
    for f in os.listdir(final_dir):
        if f.endswith('.toml') and f != "actions.toml":
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
    wait_for_images(frame)
    setpoint, basic_media, acidic_media = load_setpoints(default_setpoint, default_basic, default_acidic)

    for ch in range(1, num_channels+1):
        # find image for this frame + channel
        img_file = next(
            (f for f in os.listdir(watch_dir) if f.endswith('.png')
             and parse_filename(f) == (ch, frame)), None
        )
        if img_file is None:
            continue

        img_path = os.path.join(watch_dir, img_file)

        retries = 0
        while retries < cfg["num_tries"]:
            try:
                img = np.array(Image.open(img_path), dtype=np.float32)

                mask_file = [f for f in os.listdir(curr_mask_dir) if f.endswith(f"_channel{ch}.npy")]
                if not mask_file:
                    log(f"No mask found for channel {ch} in {curr_mask_dir}, skipping")
                    break
                mask_path = os.path.join(curr_mask_dir, mask_file[0])
                mask = np.load(mask_path) > 0

                mean_val = img[mask].mean()

                if basic_media < mean_val < acidic_media:
                    decision = decision_key['add basic media'] ### decision = decision_key['add neutral media']
                elif mean_val <= basic_media:
                    decision = decision_key['add basic media']
                else:
                    decision = decision_key['add acidic media']

                with open(os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt"), 'w') as f:
                    f.write(str(decision))

                log(f"{frame:05d}_channel{ch}: {mean_val:.3f} -> {decision_rev[decision]} "
                    f"(setpoint={setpoint:.3f}, basic={basic_media:.3f}, acidic={acidic_media:.3f})")

                break  # success

            except (OSError, ValueError, EOFError, AttributeError) as e:
                retries += 1
                log(f"Retry {retries}/{cfg['num_tries']} for frame {frame} channel {ch}: {e}")
                time.sleep(cfg["sleep_time"])

        else:
            log(f"Failed to process frame {frame} channel {ch} after {cfg['num_tries']} retries. Skipping.")

def finalize_decisions(frame):
    # Wait until all channel decisions exist
    while True:
        decs = [f for f in os.listdir(decision_dir)
                if f.startswith(f"{frame:05d}_channel") and f.endswith('.txt')]
        if len(decs) >= num_channels:
            break
        time.sleep(cfg["sleep_time"])

    # Read per-channel decisions
    decisions = []
    for ch in range(1, num_channels+1):
        dec_path = os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt")
        with open(dec_path, 'r') as f:
            decisions.append(int(f.read().strip()))

    # Define your experiment lookup table
    # This maps a combination of choices to a specific experiment name.
    # Use tuples as keys because they are hashable.
    experiment_map = {
        (decision_key['add basic media'],  decision_key['add basic media']):  "experiment1", # BASIC ALL
        (decision_key['add basic media'], decision_key['add acidic media']):  "experiment2", # BASIC 1 | ACIDIC 2
        (decision_key['add acidic media'],  decision_key['add basic media']): "experiment3", # ACIDIC 1 | BASIC 2
        (decision_key['add acidic media'], decision_key['add acidic media']): "experiment4", # ACIDIC ALL
    }

    # Access the name dynamically
    # Convert your list to a tuple to look it up
    current_combination = tuple(decisions)
    experiment_name = experiment_map.get(current_combination, "unknown_experiment")

    # Write TOML
    actions_toml = {"experiment": experiment_name}

    lock_path = os.path.join(final_dir, f"{frame}.lock")
    with open(lock_path, "w") as f:
        tomlkit.dump(actions_toml, f)

    # Rename to actions.toml
    actions_path = os.path.join(final_dir, "actions.toml")
    os.rename(lock_path, actions_path)
    log(f"Frame {frame}: wrote {actions_path} -> {experiment_name}")

# ---- INITIAL SETUP ----
log("Waiting for initial frame (0) masks and images to compute setpoint...")

while True:
    masks = [f for f in os.listdir(mask_dir) if f.startswith("000") and f.endswith('.npy')]
    if len(masks) >= num_channels:
        break
    time.sleep(cfg["sleep_time"])

initial_masks = {}
for ch in range(1, num_channels+1):
    mask_path = os.path.join(curr_mask_dir, f"00000_channel{ch}.npy")
    while not os.path.exists(mask_path):
        time.sleep(cfg["sleep_time"])
    initial_masks[ch] = np.load(mask_path) > 0

setpoint = compute_setpoint(initial_masks)
acidic_media = setpoint * 1.05
basic_media = setpoint * 0.95

save_setpoints(setpoint, basic_media, acidic_media)
log(f"Setpoint computed: {setpoint:.3f}, basic={basic_media:.3f}, acidic={acidic_media:.3f} "
    f"and saved to {setpoint_file}")

# ---- MAIN LOOP ----
frame = 0
while True:
    try:
        process_frame(frame, setpoint, basic_media, acidic_media)
        finalize_decisions(frame)
        frame += 1
    except (OSError, ValueError, EOFError, AttributeError) as e:
        log(f"Error while finalizing frame {frame}: {e}")
        time.sleep(cfg["sleep_time"])