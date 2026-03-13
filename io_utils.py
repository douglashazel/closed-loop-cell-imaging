import os
import re
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

def roi_tune_masks(watch_dir, mask_dir, curr_mask_dir, frame: int, channel: int,
                   radius=100, y_shift=0, x_shift=0):
    """
    Interactive circular ROI selector for a mask.
    Drag the circle to move it, scroll to resize it.
    Click 'Accept ROI' to zero all cells outside the circle and save
    the filtered mask to mask_dir (and curr_mask_dir if that file exists).

    Must be called from a notebook cell that begins with:  %matplotlib widget
    """
    img_name  = f"channel_{channel}_image_0_a_timepoint_{frame:05d}.png"
    mask_name = f"{frame:05d}_channel{channel}.npy"

    img_arr = io.imread(os.path.join(watch_dir, img_name))
    mask    = np.load(os.path.join(mask_dir, mask_name), allow_pickle=True)
    h, w    = img_arr.shape[:2]

    # Normalise image to uint8 RGB for overlay display
    img_f  = img_arr.astype(float)
    img_u8 = ((img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8) * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        img_u8 = np.stack([img_u8] * 3, axis=-1)

    # Precompute cell centroids
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids != 0]
    if len(cell_ids):
        cxs = np.array([np.where(mask == cid)[1].mean() for cid in cell_ids])
        cys = np.array([np.where(mask == cid)[0].mean() for cid in cell_ids])
    else:
        cxs = cys = np.array([])

    def _build_overlay(cx, cy, r):
        """RGB overlay: green = inside ROI (kept), red = outside (removed)."""
        overlay = img_u8.copy().astype(float)
        if not len(cxs):
            return overlay.astype(np.uint8), 0
        inside     = (cxs - cx)**2 + (cys - cy)**2 <= r**2
        n          = int(inside.sum())
        in_pixels  = np.isin(mask, cell_ids[inside])
        out_pixels = (mask > 0) & ~in_pixels
        overlay[in_pixels]  *= [0.3, 1.0, 0.3]
        overlay[out_pixels] *= [1.0, 0.3, 0.3]
        return overlay.astype(np.uint8), n

    def _filtered(cx, cy, r):
        if not len(cxs):
            return mask.copy(), 0
        inside = (cxs - cx)**2 + (cys - cy)**2 <= r**2
        valid  = cell_ids[inside]
        return np.where(np.isin(mask, valid), mask, 0), int(inside.sum())

    state = dict(cx=w/2 + x_shift, cy=h/2 + y_shift,
                 radius=radius, dragging=False)
    ov0, n0 = _build_overlay(state['cx'], state['cy'], state['radius'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(img_arr, cmap='gray')
    axes[0].axis('off')
    im1 = axes[1].imshow(ov0)
    axes[1].axis('off')

    circs = [
        patches.Circle((state['cx'], state['cy']), state['radius'],
                        linewidth=1.5, edgecolor='red', facecolor='none'),
        patches.Circle((state['cx'], state['cy']), state['radius'],
                        linewidth=1.5, edgecolor='red', facecolor='none'),
    ]
    for ax, c in zip(axes, circs):
        ax.add_patch(c)
    axes[0].set_title("Image with ROI  (drag=move · scroll=resize)")
    axes[1].set_title(f"{n0} cells in ROI  (green=keep · red=remove  r={state['radius']} px)")

    status = fig.text(
        0.5, 0.01,
        f"radius={state['radius']}  "
        f"y_shift={int(round(state['cy']-h/2))}  "
        f"x_shift={int(round(state['cx']-w/2))}",
        ha='center', fontsize=10, color='steelblue')
    plt.tight_layout()

    def _redraw():
        cx, cy, r = state['cx'], state['cy'], state['radius']
        ov, n = _build_overlay(cx, cy, r)
        for c in circs:
            c.center = (cx, cy)
            c.set_radius(r)
        im1.set_data(ov)
        axes[1].set_title(f"{n} cells in ROI  (green=keep · red=remove  r={r} px)")
        status.set_text(
            f"radius={r}  "
            f"y_shift={int(round(cy-h/2))}  "
            f"x_shift={int(round(cx-w/2))}")
        fig.canvas.draw_idle()

    def on_press(ev):
        if ev.inaxes in axes and ev.xdata is not None:
            state.update(dragging=True, cx=ev.xdata, cy=ev.ydata)
            _redraw()

    def on_motion(ev):
        if state['dragging'] and ev.inaxes in axes and ev.xdata is not None:
            state.update(cx=ev.xdata, cy=ev.ydata)
            _redraw()

    def on_release(ev):
        state['dragging'] = False

    def on_scroll(ev):
        if ev.inaxes in axes:
            state['radius'] = max(10, state['radius'] + (10 if ev.button == 'up' else -10))
            _redraw()

    fig.canvas.mpl_connect('button_press_event',  on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('scroll_event',         on_scroll)

    btn = widgets.Button(description='Accept ROI', button_style='success',
                         layout=widgets.Layout(width='150px'))
    out = widgets.Output()

    def on_accept(_):
        cx, cy, r = state['cx'], state['cy'], state['radius']
        filtered, n = _filtered(cx, cy, r)

        p1 = os.path.join(mask_dir, mask_name)
        np.save(p1, filtered)

        p2 = os.path.join(curr_mask_dir, mask_name)
        updated = [p1]
        if os.path.exists(p2):
            np.save(p2, filtered)
            updated.append(p2)

        with out:
            clear_output(wait=True)
            print(f"Saved ROI-filtered mask: {n} cells kept")
            for p in updated:
                print(f"  → {p}")
            print(f"  radius={r}  "
                  f"y_shift={int(round(cy-h/2))}  "
                  f"x_shift={int(round(cx-w/2))}")

    btn.on_click(on_accept)
    display(btn, out)


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