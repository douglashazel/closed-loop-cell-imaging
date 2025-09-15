import os
import json
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

# --- Load Config ---
with open("config.json", "r") as f:
    cfg = json.load(f)

mask_dir = cfg["mask_dir"]
flags_dir = cfg["flags_dir"]
check_interval = cfg["check_interval"]
num_channels = cfg["num_channels"]
threshold_ratio = cfg["threshold_ratio"]

# Track last frame ID and ROI count per channel
last_frame = defaultdict(lambda: None)
last_count = defaultdict(lambda: None)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def count_rois(mask_file):
    """Count ROIs in a Cellpose segmentation file (.npy)."""
    data = np.load(mask_file, allow_pickle=True)
    return len(np.unique(data)) - 1  # subtract background (0)

def get_frame_and_channel(filename):
    """Extract frame and channel from filename like 000_channel1.npy."""
    parts = filename.split('_channel')
    frame = int(parts[0])
    channel = int(parts[1].split('.')[0])
    return frame, channel

def create_flag_file(frame, channel, message):
    """Create a text file in flags directory with the message."""
    filename = f"{frame:03d}_channel{channel}.txt"
    filepath = os.path.join(flags_dir, filename)
    with open(filepath, 'w') as f:
        f.write(message)

log("Performance monitor started...")
while True:
    files = [f for f in os.listdir(mask_dir) if f.endswith('.npy')]
    files.sort()  # ensure chronological order

    for f in files:
        frame, channel = get_frame_and_channel(f)
        if frame is None or channel is None or channel < 1 or channel > num_channels:
            continue

        if last_frame[channel] is None or frame > last_frame[channel]:
            filepath = os.path.join(mask_dir, f)
            current_count = count_rois(filepath)

            if last_count[channel] is not None:
                prev_count = last_count[channel]
                diff = current_count - prev_count
                if diff != 0:
                    sign = '+' if diff > 0 else ''
                    msg = f"channel {channel} {'gained' if diff > 0 else 'lost'} {sign}{diff} ROI's on frame{frame}"
                    log(msg)

                    # Check for significant change (5% of previous count)
                    if prev_count > 0 and abs(diff) / prev_count >= threshold_ratio:
                        flag_msg = f"Frame: {frame}\nChannel: {channel}\nPrevious: {prev_count}\nCurrent: {current_count}\nChange: {sign}{diff}\n"
                        create_flag_file(frame, channel, flag_msg)

            last_frame[channel] = frame
            last_count[channel] = current_count

    time.sleep(check_interval)