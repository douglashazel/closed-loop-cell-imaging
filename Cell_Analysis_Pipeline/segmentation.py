import os
import numpy as np
from tqdm import tqdm
from cellpose import models, io

image_dir = 'frames'  # data path
mask_dir = 'masks'    # save path
os.makedirs(mask_dir, exist_ok=True)
images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = models.CellposeModel(gpu=True)

with tqdm(images, desc='Segmenting Images...') as pbar:
    for f in images:
        pbar.set_postfix_str(f)  # show current file name

        path = os.path.join(image_dir, f)
        img = io.imread(path)

        # Run Cellpose segmentation
        masks, _, _ = model.eval([img])
        masks = masks[0]

        # Save segmentation
        base_name = os.path.splitext(f)[0]
        save_path = os.path.join(mask_dir, base_name + '.npy')
        np.save(save_path, masks)

        pbar.update(1)