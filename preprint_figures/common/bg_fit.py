"""Background-fit primitives. Copied verbatim from april28_final_figures.py."""

import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from common.config import BG_FIT
from common.io_paths import channel_dir, load_segmentation, sorted_image_files


def poly_design(xn, yn, degree):
    """Build a 2-D polynomial design matrix up to total degree ``degree``."""
    xn = np.asarray(xn, dtype=np.float64)
    yn = np.asarray(yn, dtype=np.float64)
    cols = [
        (xn ** i) * (yn ** j)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]
    return np.stack(cols, axis=-1)


def build_background_sampler(cfg, fit_cfg=BG_FIT):
    """Build a reusable background sampler for one experiment.

    Returns a dict containing:
        shape          - (H, W) of the frame
        sample_points  - Nx2 array of (x, y) sample-point coords
        A_pinv         - pseudoinverse of the design matrix at the sample points
        A_min          - design matrix on a coarse grid for finding the fit minimum
        patch_specs    - precomputed circular-patch slicers for each sample point
        degree         - polynomial degree
        clearance      - HxW distance-to-nearest-cell-pixel map (debug aid)
    """
    first_dir = os.path.join(channel_dir(cfg, cfg["channels"][0]), "frames")
    first_files = sorted_image_files(first_dir)
    H, W = np.array(Image.open(os.path.join(first_dir, first_files[0])).convert("L")).shape

    union = np.zeros((H, W), dtype=bool)
    for ch in cfg["channels"]:
        mdir = os.path.join(channel_dir(cfg, ch), "masks")
        mfiles = sorted([f for f in os.listdir(mdir) if f.endswith(".npy")])
        if not mfiles:
            continue
        idxs = np.unique(
            np.linspace(
                0,
                len(mfiles) - 1,
                min(fit_cfg["n_mask_snapshots"], len(mfiles)),
                dtype=int,
            )
        )
        for i in idxs:
            mask = load_segmentation(os.path.join(mdir, mfiles[i])) > 0
            if mask.shape != (H, W):
                raise ValueError(
                    f"{ch} mask shape {mask.shape} does not match frame shape {(H, W)}"
                )
            union |= mask

    clearance = distance_transform_edt(~union)

    margin = fit_cfg["sample_r"]
    search_r = fit_cfg["search_r"]
    sample_points = []
    for gy in np.linspace(margin, H - margin - 1, fit_cfg["grid_n"]).astype(int):
        for gx in np.linspace(margin, W - margin - 1, fit_cfg["grid_n"]).astype(int):
            y0, y1 = max(margin, gy - search_r), min(H - margin, gy + search_r + 1)
            x0, x1 = max(margin, gx - search_r), min(W - margin, gx + search_r + 1)
            sub = clearance[y0:y1, x0:x1]
            idx = np.unravel_index(sub.argmax(), sub.shape)
            if sub[idx] >= fit_cfg["clearance_r"]:
                sample_points.append((x0 + idx[1], y0 + idx[0]))
    sample_points = np.asarray(sample_points, dtype=int)

    n_terms = (fit_cfg["poly_degree"] + 1) * (fit_cfg["poly_degree"] + 2) // 2
    if len(sample_points) < n_terms:
        raise ValueError(
            f"Only {len(sample_points)} cell-free background samples found; "
            f"need at least {n_terms}."
        )

    xn_samples = (sample_points[:, 0] - W / 2) / (W / 2)
    yn_samples = (sample_points[:, 1] - H / 2) / (H / 2)
    A_samples = poly_design(xn_samples, yn_samples, fit_cfg["poly_degree"])
    A_pinv = np.linalg.pinv(A_samples)

    g = fit_cfg["min_grid_n"]
    gx = np.linspace(-1, 1, g)
    gy = np.linspace(-1, 1, g)
    GX, GY = np.meshgrid(gx, gy)
    A_min = poly_design(GX.ravel(), GY.ravel(), fit_cfg["poly_degree"])

    r = fit_cfg["sample_r"]
    patch_specs = []
    for sx, sy in sample_points:
        y0, y1 = max(0, sy - r), min(H, sy + r + 1)
        x0, x1 = max(0, sx - r), min(W, sx + r + 1)
        py = np.arange(y0, y1)[:, None]
        px = np.arange(x0, x1)[None, :]
        cmask = (py - sy) ** 2 + (px - sx) ** 2 <= r ** 2
        patch_specs.append((y0, y1, x0, x1, cmask))

    return {
        "shape": (H, W),
        "sample_points": sample_points,
        "A_pinv": A_pinv,
        "A_min": A_min,
        "patch_specs": patch_specs,
        "degree": fit_cfg["poly_degree"],
        "clearance": clearance,
    }


def sample_background_means(img, sampler):
    """Return the mean intensity inside each cell-free sample patch on ``img``."""
    means = np.empty(len(sampler["patch_specs"]), dtype=np.float64)
    for pi, (y0, y1, x0, x1, cmask) in enumerate(sampler["patch_specs"]):
        means[pi] = img[y0:y1, x0:x1][cmask].mean()
    return means


def fit_background_from_image(img, sampler):
    """Fit the polynomial background to one frame.

    Returns ``(coefs, bg_min, means)`` where ``bg_min`` is the minimum of the
    fitted surface over a coarse grid (subtracted later so the corrected image
    sits at zero baseline).
    """
    means = sample_background_means(img, sampler)
    coefs = sampler["A_pinv"] @ means
    bg_min = float((sampler["A_min"] @ coefs).min())
    return coefs, bg_min, means


def eval_background_at_points(coefs, bg_min, xs, ys, shape, degree):
    """Evaluate the fitted background surface at arbitrary ``(x, y)`` pixels."""
    H, W = shape
    xn = (np.asarray(xs, dtype=np.float64) - W / 2) / (W / 2)
    yn = (np.asarray(ys, dtype=np.float64) - H / 2) / (H / 2)
    return poly_design(xn, yn, degree) @ coefs - bg_min


def eval_background_image(coefs, bg_min, shape, degree):
    """Render the fitted background as a full HxW image."""
    H, W = shape
    xn = (np.arange(W, dtype=np.float32) - W / 2) / (W / 2)
    yn = (np.arange(H, dtype=np.float32) - H / 2) / (H / 2)
    XF, YF = np.meshgrid(xn, yn)
    bg = np.zeros((H, W), dtype=np.float32)
    k = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            bg += coefs[k] * (XF ** i) * (YF ** j)
            k += 1
    return bg - bg_min
