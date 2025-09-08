import os
import time
import numpy as np
from datetime import datetime
from cellpose import models, io

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

log("Cellpose model loading complete.")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

watch_dir = 'incoming_frames'
mask_dir = 'processed_masks'
for d in [watch_dir, mask_dir]:
    os.makedirs(d, exist_ok=True)

# Track processed images by checking existing .npy files
existing_masks = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.endswith('.npy')}
processed = {os.path.join(watch_dir, f) for f in os.listdir(watch_dir) if os.path.splitext(f)[0] in existing_masks}

model = models.CellposeModel(gpu=True)

while True:
    images = sorted([f for f in os.listdir(watch_dir) if f.endswith(('.png', '.jpg'))])
    for f in images:
        path = os.path.join(watch_dir, f)
        if path not in processed:
            log(f'Processing {f}...')
            processed.add(path)
            start_time = time.time()

            # Load image
            img = io.imread(path)

            # Run Cellpose segmentation
            masks, _, _ = model.eval([img])
            masks = masks[0]

            # Save segmentation
            base_name = os.path.splitext(f)[0]
            save_path = os.path.join(mask_dir, base_name + '.npy')
            np.save(save_path, masks)

            elapsed = time.time() - start_time
            log(f'Done {f} in {elapsed:.2f} seconds')

    time.sleep(2)