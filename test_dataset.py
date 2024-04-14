from dataset import MPEG4, clip_and_scale
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os


filenames = Path('.').glob('*avi')
filenames = list(filenames)[:1]

dataset = MPEG4(filenames, accumulate=True, GOP_SIZE=12)
loader = DataLoader(dataset, shuffle=False)

for i, d in enumerate(iter(MPEG4)):
    print(f"[{d['filename']}] [{d['num_frames']}]")
    for k, v in d.items():
        print(f'{k}:', 'max:', v.max(), 'min:', v.min, 'shape:', v.shape, v.dtype)
        img = Image.fromarray(v)
        if k == dataset.MV:
            img = clip_and_scale(img, 20)
            img += 128
            img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
        elif k == dataset.RES:
            img += 128
            img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
        elif k == dataset.IFRAME:
            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img = img[..., ::-1]
        img.save(f"{os.path.basename(d['filename'])[:-4]}_idx{i}_{k}.png")
