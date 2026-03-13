import os
import re
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from cellpose import models, io, utils
from scipy.ndimage import binary_dilation
import ipywidgets as widgets
from IPython.display import display, clear_output

def parse_filename(fname):
    """Extract frame and channel from filename like channel_1_image_0_a_timepoint_00000.png"""
    base = os.path.splitext(fname)[0]
    m = re.search(r'channel_(\d+).*timepoint_(\d+)$', base)
    if not m:
        raise ValueError(f"Unexpected filename format: {fname}")
    channel = int(m.group(1))
    frame = int(m.group(2))
    return frame, channel, base

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
            frame, channel, base_name = parse_filename(f)
            base_name = f"{frame:05d}_channel{channel}"

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
            masks, _, _ = model.eval(img, **eval_kwargs)

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
            fig = plt.figure(figsize=(10, 5), dpi=300)
            # Left = overlay
            plt.subplot(1, 2, 1)
            plt.imshow(overlay)
            plt.title("With segmentation")
            plt.axis("off")

            # Right = raw
            plt.subplot(1, 2, 2)
            plt.imshow(img, cmap="gray")
            plt.title("Raw image")
            plt.axis("off")

            save_path = os.path.join(temp_overlays, f"{base_name}_overlay.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)  # free memory
            pbar.update(1)

def visualize_segmentation(watch_dir, mask_dir, frame: int, channel: int):
    img_name = f"channel_{channel}_image_0_a_timepoint_{frame:05d}.png"
    mask_name = f"{frame:05d}_channel{channel}.npy"

    img_path = os.path.join(watch_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)

    img = io.imread(img_path)
    masks = np.load(mask_path, allow_pickle=True)

    outlines = utils.masks_to_outlines(masks)
    outlines = binary_dilation(outlines, iterations=2)

    overlay = img.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)
    overlay[outlines] = [255, 0, 0]

    raw_button = widgets.Button(description="Show Raw")
    seg_button = widgets.Button(description="Show Segmentation")

    out = widgets.Output()

    def show_raw(b):
        with out:
            clear_output(wait=True)
            plt.figure(figsize=(12, 12))
            plt.imshow(img, cmap="gray")
            plt.title(f"Raw Frame {frame:05d}, Channel {channel}")
            plt.axis("off")
            plt.show()

    def show_seg(b):
        with out:
            clear_output(wait=True)
            plt.figure(figsize=(12, 12))
            plt.imshow(overlay)
            plt.title(f"Segmentation Frame {frame:05d}, Channel {channel}")
            plt.axis("off")
            plt.show()

    raw_button.on_click(show_raw)
    seg_button.on_click(show_seg)

    display(widgets.HBox([raw_button, seg_button]))
    display(out)
    show_raw(None)  # default

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