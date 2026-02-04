import napari
import numpy as np
from tqdm import tqdm
import imageio.v3 as iio
from pathlib import Path
from basicpy import BaSiC

folder = Path("DMSO_C2C12_repeat_pulse_16JAN26_take2/channel 1/frames")
ext = "*.png"

files = sorted(folder.glob(ext))

imgs = []
for f in tqdm(files):
    img = iio.imread(f)
    if img.ndim == 3:
        img = img[..., 0]
    imgs.append(img)

stack = np.stack(imgs, axis=0).astype(np.float32)

model = BaSiC(get_darkfield=False, smoothness_flatfield=1.0, smoothness_darkfield=0.0)
model.fit(stack)

corrected = model.transform(stack)

flatfield = model.flatfield

viewer = napari.Viewer()

viewer.add_image(stack, name="raw", contrast_limits=(np.percentile(stack,1), np.percentile(stack,99)))
viewer.add_image(flatfield, name="flatfield")
viewer.add_image(corrected, name="corrected", contrast_limits=(np.percentile(corrected,1), np.percentile(corrected,99)))

napari.run()