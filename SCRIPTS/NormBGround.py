import os
import re
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from io_utils import load_msgpack, save_msgpack, lum_dict_to_df

def extract_number(filename):
    match = re.search(r'timepoint_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_image(path):
    return np.array(Image.open(path))

def compute_bground(images, image_dir, mask_dir):
    bg_trace = []
    for fname in tqdm(images, desc="Background"):
        img = load_image(os.path.join(image_dir, fname))
        if img.ndim == 3:
            img = img.mean(axis=-1)
        mask = np.load(os.path.join(mask_dir, os.path.splitext(fname)[0] + ".npy"))
        bg_trace.append(img[mask == 0].mean())
    return np.array(bg_trace)

def process_tag(tag, images, bg_trace, analysis_dir):
    lum_json_path = os.path.join(analysis_dir, f"luminosity{tag}.json")
    lum_df = lum_dict_to_df(load_msgpack(lum_json_path))

    # normalize by subtracting per-frame background
    norm_df = lum_df.copy()
    for i, _ in enumerate(images):
        col = f"f{i}"
        if col in norm_df.columns:
            norm_df[col] = norm_df[col] - bg_trace[i]

    # drop background cell row
    norm_df = norm_df[norm_df["CellID"] != 0]

    # save as msgpack JSON
    norm_dict = {
        str(int(row['CellID'])): {col: row[col] for col in norm_df.columns if col != 'CellID'}
        for _, row in norm_df.iterrows()
    }
    norm_json_path = os.path.join(analysis_dir, f"luminosity_no_bground{tag}.json")
    save_msgpack(norm_dict, norm_json_path)

    # plot
    cmap = plt.get_cmap("twilight_shifted")
    colors = cmap(np.linspace(0, 1, len(norm_df)))

    plt.figure(dpi=300)
    for (_, row), color in zip(norm_df.iterrows(), colors):
        non_nan = row.dropna()
        frames = [int(str(x).lstrip("f")) for x in non_nan.index if str(x).startswith("f")]
        vals = non_nan[[c for c in non_nan.index if str(c).startswith("f")]].values
        plt.plot(frames, vals, alpha=0.7, color=color)

    plt.xlabel("Frame")
    plt.ylabel("Average luminosity (cell - background)")
    plt.title("Cell luminosity over time")

    plot_dir = os.path.join(analysis_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig_path = os.path.join(plot_dir, f"average_luminosity_no_bground{tag}.png")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return norm_json_path, fig_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--analysis_dir", required=True)
    args = parser.parse_args()

    image_dir = args.image_dir
    mask_dir = args.mask_dir
    analysis_dir = args.analysis_dir

    images = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))],
        key=extract_number
    )
    bg_trace = compute_bground(images, image_dir, mask_dir)

    for tag in ['_complete', '']:
        process_tag(tag, images, bg_trace, analysis_dir)
