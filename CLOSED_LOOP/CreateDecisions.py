import os
import time
import json
import tomlkit
import numpy as np
from PIL import Image

from io_utils import log, load_config, parse_filename

cfg = load_config()
num_channels    = cfg["num_channels"]
watch_dir       = cfg["watch_dir"]
curr_mask_dir   = cfg["curr_mask_dir"]
decision_dir    = cfg["decision_dir"]
final_dir       = cfg["final_dir"]
setpoint_file   = cfg["setpoint_file"]
luminosity_file = cfg["luminosity_file"]

decision_key    = cfg['decision_key']
decision_rev    = {v: k for k, v in decision_key.items()}
continuous_seg  = cfg.get("continuous_segmentation", True)

# Tracks the active mask filename per channel so we can log hot-swaps exactly once
_active_mask_name = {}

def append_luminosity(frame, channel, mean_val, setpoint, decision_label):
    """Append a luminosity record to the JSON log file."""
    record = {
        "frame": frame,
        "channel": channel,
        "mean_luminosity": round(float(mean_val), 4),
        "setpoint": round(float(setpoint), 4),
        "decision": decision_label,
    }
    # Load existing data or start fresh
    if os.path.exists(luminosity_file):
        with open(luminosity_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open(luminosity_file, 'w') as f:
        json.dump(data, f)

def get_latest_complete_frame():
    """Return the highest frame number that has all channels present, or -1."""
    frame_counts = {}
    for f in os.listdir(watch_dir):
        if not f.endswith('.png'):
            continue
        parsed = parse_filename(f)
        if parsed is None:
            continue
        _, frame_num = parsed
        frame_counts[frame_num] = frame_counts.get(frame_num, 0) + 1
    complete = [fr for fr, count in frame_counts.items() if count >= num_channels]
    return max(complete) if complete else -1

def wait_for_frame(frame):
    """Block until the given frame has all channels present."""
    while True:
        imgs = [f for f in os.listdir(watch_dir)
                if f.endswith('.png') and parse_filename(f)[1] == frame]
        if len(imgs) >= num_channels:
            return
        time.sleep(cfg["sleep_time"])

def compute_setpoint(initial_masks):
    setpoint_vals = []
    for ch in range(1, num_channels + 1):
        img_file = next(
            (f for f in os.listdir(watch_dir)
             if f.endswith('.png') and parse_filename(f) == (ch, 0)), None
        )
        if img_file is None:
            continue
        img = np.array(Image.open(os.path.join(watch_dir, img_file)), dtype=np.float32)
        setpoint_vals.append(img[initial_masks[ch]].mean())
    # Scale 5% below the mean initial fluorescence as the working setpoint
    return np.mean(setpoint_vals) * 0.95

def save_setpoints(setpoint):
    with open(setpoint_file, 'w') as f:
        f.write(f"setpoint={setpoint:.6f}\n")

def load_setpoints(default_setpoint):
    if not os.path.exists(setpoint_file):
        return default_setpoint
    try:
        with open(setpoint_file, 'r') as f:
            for line in f:
                if line.startswith('setpoint='):
                    return float(line.strip().split('=', 1)[1])
    except Exception as e:
        log(f"Warning: failed to parse {setpoint_file}: {e}")
    return default_setpoint

def process_frame(frame, initial_masks, default_setpoint):
    """
    Evaluate a single frame.
    - continuous_segmentation=False: uses fixed in-memory frame-0 masks (no disk reads).
    - continuous_segmentation=True:  re-reads curr_mask_dir each frame, so any mask the user
      pushes from preprocess.ipynb mid-experiment is picked up automatically (hot-swap).
    """
    wait_for_frame(frame)
    setpoint = load_setpoints(default_setpoint)

    for ch in range(1, num_channels + 1):
        img_file = next(
            (f for f in os.listdir(watch_dir)
             if f.endswith('.png') and parse_filename(f) == (ch, frame)), None
        )
        if img_file is None:
            continue

        img_path = os.path.join(watch_dir, img_file)

        for attempt in range(cfg["num_tries"]):
            try:
                img = np.array(Image.open(img_path), dtype=np.float32)

                if continuous_seg:
                    mask_files = [f for f in os.listdir(curr_mask_dir)
                                  if f.endswith(f"_channel{ch}.npy")]
                    if mask_files:
                        mask_fname = mask_files[0]
                        if _active_mask_name.get(ch) != mask_fname:
                            log(f"Channel {ch}: active mask -> '{mask_fname}'")
                            _active_mask_name[ch] = mask_fname
                        mask = np.load(os.path.join(curr_mask_dir, mask_fname)) > 0
                    else:
                        mask = initial_masks[ch]
                else:
                    mask = initial_masks[ch]

                mean_val = img[mask].mean()

                if mean_val >= setpoint:
                    decision = decision_key['add acidic media']
                else:
                    decision = decision_key['add neutral media']

                with open(os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt"), 'w') as f:
                    f.write(str(decision))

                append_luminosity(frame, ch, mean_val, setpoint, decision_rev[decision])
                log(f"{frame:05d}_channel{ch}: {mean_val:.3f} -> {decision_rev[decision]} "
                    f"(setpoint={setpoint:.3f})")
                break

            except (OSError, ValueError, EOFError, AttributeError, SyntaxError) as e:
                log(f"Retry {attempt + 1}/{cfg['num_tries']} for frame {frame} channel {ch}: {e}")
                time.sleep(1)
        else:
            log(f"Failed frame {frame} channel {ch} after {cfg['num_tries']} retries. Skipping.")

def finalize_decisions(frame):
    # Wait until all channel decisions are written
    while True:
        decs = [f for f in os.listdir(decision_dir)
                if f.startswith(f"{frame:05d}_channel") and f.endswith('.txt')]
        if len(decs) >= num_channels:
            break
        time.sleep(cfg["sleep_time"])

    decisions = []
    for ch in range(1, num_channels + 1):
        with open(os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt"), 'r') as f:
            decisions.append(int(f.read().strip()))

    # Map per-channel decision tuple to experiment name via config
    experiment_map = cfg.get("experiment_map", {})
    experiment_name = experiment_map.get(str(tuple(decisions)), "unknown_experiment")

    # Write atomically via lock file then rename
    lock_path    = os.path.join(final_dir, f"{frame:05d}.lock")
    actions_path = os.path.join(final_dir, "actions.toml")
    with open(lock_path, "w") as f:
        tomlkit.dump({"experiment": experiment_name}, f)
    os.rename(lock_path, actions_path)

    log(f"Frame {frame}: wrote actions.toml -> {experiment_name}")

    # Clean up per-channel decision files (already logged above)
    for ch in range(1, num_channels + 1):
        dec_path = os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt")
        try:
            os.remove(dec_path)
        except FileNotFoundError:
            pass

# ---- INITIAL SETUP ----
# Stall here until the user has run preprocess.ipynb and executed the
# "Update Masks" cell, which copies the approved frame-0 segmentation
# into curr_mask_dir.  Nothing starts until the user is happy.
log("Waiting for user to push approved frame-0 masks via preprocess.ipynb...")

for ch in range(1, num_channels + 1):
    mask_path = os.path.join(curr_mask_dir, f"00000_channel{ch}.npy")
    while not os.path.exists(mask_path):
        time.sleep(cfg["sleep_time"])

initial_masks = {
    ch: np.load(os.path.join(curr_mask_dir, f"00000_channel{ch}.npy")) > 0
    for ch in range(1, num_channels + 1)
}

setpoint = compute_setpoint(initial_masks)

save_setpoints(setpoint)
log(f"Setpoint computed: {setpoint:.3f} (95% of initial brightness) — saved to {setpoint_file}")

# ---- MAIN LOOP ----
# Process every frame sequentially (log luminosity for each), but only
# finalize a decision when this is the latest available frame.  This way
# intermediate images captured during a pulse are observed but don't
# trigger new actions.
frame = 0
while True:
    try:
        process_frame(frame, initial_masks, setpoint)
        latest = get_latest_complete_frame()
        if frame >= latest:
            finalize_decisions(frame)
        else:
            log(f"Frame {frame}: skipping decision (latest frame is {latest})")
            # Clean up decision files for this skipped frame
            for ch in range(1, num_channels + 1):
                dec_path = os.path.join(decision_dir, f"{frame:05d}_channel{ch}.txt")
                try:
                    os.remove(dec_path)
                except FileNotFoundError:
                    pass
        frame += 1
    except (OSError, ValueError, EOFError, AttributeError, SyntaxError) as e:
        log(f"Error on frame {frame}: {e}")
        time.sleep(cfg["sleep_time"])