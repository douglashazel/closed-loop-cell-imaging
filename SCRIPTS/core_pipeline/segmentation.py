import os
import argparse
import numpy as np
from tqdm import tqdm
from cellpose import models, io

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True)
parser.add_argument("--mask_dir", required=True)
parser.add_argument("--flow_threshold", type=float, default=0.4)
parser.add_argument("--cellprob_threshold", type=float, default=0.0)
parser.add_argument("--niter", type=int, default=200)
parser.add_argument("--diameter", type=int, default=0,
                    help="Cell diameter in pixels (0 = let Cellpose estimate it)")
parser.add_argument("--cpu", action="store_true",
                    help="Run Cellpose on the CPU (default: CUDA when available)")
args = parser.parse_args()

image_dir = args.image_dir
mask_dir = args.mask_dir
os.makedirs(mask_dir, exist_ok=True)
images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

# Cellpose divides by the diameter, so 0 is passed as None (auto-estimate),
# the same mapping the TUNE_GUI preview uses.
diameter = args.diameter if args.diameter > 0 else None

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
model = models.CellposeModel(gpu=not args.cpu)


def mask_is_readable(path):
    """True when ``path`` is a complete .npy file (a killed run can leave a truncated one)."""
    try:
        np.load(path, mmap_mode="r")
        return True
    except (ValueError, OSError, EOFError):
        return False


with tqdm(images, desc='Segmenting Images...') as pbar:
    for f in images:
        base_name = os.path.splitext(f)[0]
        save_path = os.path.join(mask_dir, base_name + '.npy')
        if os.path.exists(save_path):
            if mask_is_readable(save_path):
                continue
            print(f"\nRegenerating unreadable mask {save_path}")

        pbar.set_postfix_str(f)

        path = os.path.join(image_dir, f)
        img = io.imread(path)

        masks, _, _ = model.eval(
            [img],
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            niter=args.niter,
            diameter=diameter,
        )
        masks = masks[0]
        # Write under a temporary name, then rename: trajectories.py polls for
        # <base>.npy and must never load a half-written mask.
        tmp_path = save_path + '.tmp'
        with open(tmp_path, 'wb') as fh:
            np.save(fh, masks)
        os.replace(tmp_path, save_path)

        pbar.update(1)
