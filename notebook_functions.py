import os
import re
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cellpose import models, io, utils
from scipy.ndimage import binary_dilation

def tune_masks(unique_params, watch_dir, mask_dir, temp_overlays):
    """
    Segment images in a directory using Cellpose with channel-specific parameters, save masks, 
    and generate overlay images highlighting segmentation outlines.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = models.CellposeModel(gpu=True)

    images = sorted([f for f in os.listdir(watch_dir) if f.endswith(('.png', '.jpg'))])
    with tqdm(images, desc='Segmenting Images...') as pbar:
        for f in images:
            base_name = os.path.splitext(f)[0]

            pbar.set_postfix_str(f)
            path = os.path.join(watch_dir, f)

            img = io.imread(path)

            # parse channel number from filename like "channel_1_image_0_a_timepoint_00000.png"
            match = re.search(r"channel_(\d+)", f)
            if match:
                channel = f"channel{match.group(1)}"
            else:
                channel = None

            # skip if channel not in active params
            if channel not in unique_params:
                pbar.update(1)
                continue

            # pick params
            eval_kwargs = {k: v for k, v in unique_params[channel].items() if v is not None}

            # run segmentation with channel-specific params
            masks, flows, styles = model.eval(img, **eval_kwargs)

            # save masks
            mask_path = os.path.join(mask_dir, f"{base_name}.npy")
            np.save(mask_path, masks)

            # outlines
            outlines = utils.masks_to_outlines(masks)
            outlines = binary_dilation(outlines, iterations=3)
            overlay = img.copy()
            if overlay.ndim == 2:
                overlay = np.stack([overlay] * 3, axis=-1)
            overlay[outlines] = [255, 0, 0]

            # save overlay
            fig = plt.figure(dpi=300)
            plt.title(base_name)
            plt.imshow(overlay)
            plt.axis("off")
            save_path = os.path.join(temp_overlays, f"{base_name}_overlay.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)  # free memory
            pbar.update(1)


def visualize_segmentation(watch_dir, mask_dir, frame: int, channel: int):
    """
    Display the image for a given frame and channel with segmentation outlines overlaid.
    Expects filenames like: <frame:05d>_channel<channel>.png
    """
    # Build filenames to match new convention
    img_name = f"{frame:05d}_channel{channel}.png"
    mask_name = os.path.splitext(img_name)[0] + ".npy"

    img_path = os.path.join(watch_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Load
    img = io.imread(img_path)
    masks = np.load(mask_path, allow_pickle=True)
    
    # Make outlines
    outlines = utils.masks_to_outlines(masks)
    outlines = binary_dilation(outlines, iterations=2)
    
    # Overlay
    overlay = img.copy()
    if overlay.ndim == 2:  # grayscale
        overlay = np.stack([overlay] * 3, axis=-1)
    overlay[outlines] = [255, 0, 0]
    
    # Plot
    plt.figure(dpi=300)
    plt.imshow(overlay, cmap="gray")
    plt.title(f"Frame {frame:05d}, Channel {channel}")
    plt.axis("off")
    plt.show()


def update_masks(channel_updates, mask_dir, curr_mask_dir):
    """
    Replace old masks in the current directory with updated masks for specified channels.
    """

    for channel, frame in channel_updates.items():
        src = os.path.join(mask_dir, f"{frame}_{channel}.npy")
        dst = os.path.join(curr_mask_dir, f"{frame}_{channel}.npy")

        if os.path.exists(src):
            # remove any old masks for this channel in current_masks
            for f in os.listdir(curr_mask_dir):
                if f.endswith(f"_{channel}.npy"):
                    os.remove(os.path.join(curr_mask_dir, f))

            # copy new mask into place
            shutil.copy2(src, dst)
            print(f"Updated {channel} mask → {dst}")
        else:
            print(f"Warning: {src} not found.")