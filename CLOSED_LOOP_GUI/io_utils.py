"""
Shared utilities for the closed-loop pipeline.
All scripts import log(), load_config(), and parse_filename() from here.
"""
import os
import re
import json
from datetime import datetime

# Matches filenames like: channel_1_image_0_a_timepoint_00000.png (.png or
# .jpg/.jpeg, any case). Every script filters frames through parse_filename().
_FILENAME_PATTERN = re.compile(r"channel_(\d+).*timepoint_(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def parse_filename(fname):
    """Return (channel, frame) ints from an image filename, or (None, None) on no match."""
    m = _FILENAME_PATTERN.search(os.path.basename(fname))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))
