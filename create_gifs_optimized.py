import os
import re
import shutil
import msgpack
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from skimage.measure import find_contours

def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_json(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return msgpack.unpack(f, raw=False)
    return {}

def process_cell(args):
    cell_id, coords, png_files, tif_path, mask_dir, save_path, movie, FIXED_CROP_SIZE = args

    movie_safe = movie.replace("/", "_")
    output_dir = f"{save_path}/temp_images_{cell_id}"
    os.makedirs(output_dir, exist_ok=True)
    image_files = []

    # pick fixed anchor location = first valid coordinate
    anchor_x, anchor_y = None, None
    for frame_idx in range(len(png_files)):
        xkey = f"x{frame_idx}"
        ykey = f"y{frame_idx}"

        if xkey in coords and ykey in coords and coords[xkey] is not None and coords[ykey] is not None:
            anchor_x = int(coords[xkey])
            anchor_y = int(coords[ykey])
            break

    if anchor_x is None or anchor_y is None:
        print(f"No valid coordinates for Cell {cell_id}")
        shutil.rmtree(output_dir)
        return

    for frame_idx, png_file in enumerate(png_files):
        xkey = f"x{frame_idx}"
        ykey = f"y{frame_idx}"
        curr_x = int(coords[xkey]) if coords.get(xkey) is not None else anchor_x
        curr_y = int(coords[ykey]) if coords.get(ykey) is not None else anchor_y

        tif_image = plt.imread(os.path.join(tif_path, png_file))
        base_name = os.path.splitext(png_file)[0]
        mask_path = os.path.join(mask_dir, base_name + ".npy")

        contours = []
        if os.path.exists(mask_path):
            mask_all = np.load(mask_path, allow_pickle=True)
            if 0 <= curr_y < mask_all.shape[0] and 0 <= curr_x < mask_all.shape[1]:
                mask_id = mask_all[curr_y, curr_x]
                if mask_id > 0:
                    mask = (mask_all == mask_id).astype(np.uint8)
                    contours = find_contours(mask, level=0.5)

        half_width = FIXED_CROP_SIZE // 2
        img_height, img_width = tif_image.shape[:2]
        crop_min_y = max(0, anchor_y - half_width)
        crop_max_y = min(img_height, anchor_y + half_width)
        crop_min_x = max(0, anchor_x - half_width)
        crop_max_x = min(img_width, anchor_x + half_width)

        cropped_image = tif_image[crop_min_y:crop_max_y, crop_min_x:crop_max_x]

        if cropped_image.shape[0] < FIXED_CROP_SIZE or cropped_image.shape[1] < FIXED_CROP_SIZE:
            padded_image = np.zeros((FIXED_CROP_SIZE, FIXED_CROP_SIZE), dtype=cropped_image.dtype)
            y_offset = (FIXED_CROP_SIZE - cropped_image.shape[0]) // 2
            x_offset = (FIXED_CROP_SIZE - cropped_image.shape[1]) // 2
            padded_image[y_offset:y_offset + cropped_image.shape[0],
                         x_offset:x_offset + cropped_image.shape[1]] = cropped_image
            cropped_image = padded_image

        plt.figure(dpi=300)
        plt.title(f"Frame{frame_idx}")
        plt.imshow(cropped_image, cmap='gray', alpha=1)
        for contour in contours:
            cropped_contour_y = contour[:, 0] - crop_min_y
            cropped_contour_x = contour[:, 1] - crop_min_x
            plt.plot(cropped_contour_x, cropped_contour_y, color='red', linewidth=.5)

        plt.axis('off')
        output_file = f"{output_dir}/{movie_safe}_{cell_id}_frame_{frame_idx:03d}.png"
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
        image_files.append(output_file)

    gif_path = f"{save_path}/{movie_safe}_{cell_id}.gif"
    with imageio.get_writer(gif_path, mode='I', duration=1, loop=0) as writer:
        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

    print(f"GIF saved at: {gif_path}")
    shutil.rmtree(output_dir)

# ----- CHANGE HERE ----- #
movie = "DMSO_C2C12_repeat_pulse_16JAN26_take2/channel_1_edited"
FIXED_CROP_SIZE = 200
save_path = "gifs"
NUM_WORKERS = 5

tif_path = f"{movie}/frames"
mask_dir = f"{movie}/masks"
save_cell_path = f"{movie}/analysis"
traj_path = f"{save_cell_path}/trajectories_complete.json"

if not os.path.exists(traj_path):
    print(f"Missing trajectories file: {traj_path}")
    exit()
if not os.path.exists(tif_path):
    print(f"Missing movie frames: {tif_path}")
    exit()

png_files = sorted([f for f in os.listdir(tif_path) if f.endswith('.png')], key=extract_number)
traj_dict = load_json(traj_path)
os.makedirs(save_path, exist_ok=True)

cell_ids = [val for idx, val in enumerate(traj_dict.keys()) if idx<=4]

args_list = [(cell_id, traj_dict[cell_id], png_files, tif_path, mask_dir, save_path, movie, FIXED_CROP_SIZE) for cell_id in cell_ids]

if __name__ == "__main__":
    with Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_cell, args_list), total=len(args_list), desc="Processing cells"))