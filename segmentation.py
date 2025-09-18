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
args = parser.parse_args()

image_dir = args.image_dir
mask_dir = args.mask_dir
os.makedirs(mask_dir, exist_ok=True)
images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = models.CellposeModel(gpu=True)

with tqdm(images, desc='Segmenting Images...') as pbar:
    for f in images:
        pbar.set_postfix_str(f)

        path = os.path.join(image_dir, f)
        img = io.imread(path)

        masks, _, _ = model.eval(
            [img],
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            niter=args.niter,
            diameter=None
        )
        masks = masks[0]

        base_name = os.path.splitext(f)[0]
        save_path = os.path.join(mask_dir, base_name + '.npy')
        np.save(save_path, masks)

        pbar.update(1)