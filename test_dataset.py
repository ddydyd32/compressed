from dataset import MPEG4, clip_and_scale
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from coviar import load as load_mpeg4

import os
os.makedirs('t', exist_ok=True)

filenames = Path('/home/v-dongyaozhu/compressed/pytorch-coviar/data/hmdb51/mpeg4_videos/brush_hair/').glob('*mp4')
filenames = list(sorted(map(str, filenames)))[:1]

dataset = MPEG4(filenames, accumulate=True, GOP_SIZE=12)
loader = DataLoader(dataset, shuffle=False)

for i, d in enumerate(iter(dataset)):
    if i > 4:
        break
    print(f"\n[{i} / {d['num_frames']}]")
    for k, v in d.items():
        if type(v) in [str, int]:
            continue
        v2 = v
        v = v['tensor'].cpu().numpy()
        print(f"{k}[{v2['gop_index']}, {v2['gop_pos']}]:", 'min:', v.min(), 'max:', v.max(), 'mean:', v.mean(), 'shape:', v.shape, v.dtype)
        d[k] = v
    d['img'] = load_mpeg4(d['filename'], d['gop_index'], d['gop_pos'], 0, True)
    img = d['img'][..., ::-1]
    iframe = d['iframe'][..., ::-1]
    mv = d['mv']
    res = d['res'][..., ::-1]
    height, width, _ = img.shape
    print('diff:', iframe.mean() + res.mean())

    yv, xv = np.meshgrid(range(width), range(height))
    dx, dy = mv[:, :, 0], mv[:, :, 1]
    if i % 12 == 0:
        assert np.abs(iframe[xv, yv] - img).sum() < 1e-5
        assert np.abs(res).sum() < 1e-5
        assert np.abs(mv).sum() < 1e-5
    src_x, src_y = xv - dx, yv - dy
    src_x = np.clip(src_x, 0, height-1)
    src_y = np.clip(src_y, 0, width-1)
    move = iframe[src_x, src_y].astype(np.float32)
    print('move:', 'min:', move.min(), 'max:', move.max(), 'mean:', move.mean(), move.dtype, move.shape)
    recons = move + res.astype(np.float32)
    print('recons:', 'min:', recons.min(), 'max:', recons.max(), 'mean:', recons.mean(), recons.dtype, recons.shape)
    move = np.clip(move, 0, 255).astype(np.uint8)
    recons = np.clip(recons, 0, 255).astype(np.uint8)
    Image.fromarray(move).save(f"t/idx_{i}_move_{os.path.basename(d['filename'])[:-4]}.png")
    Image.fromarray(recons).save(f"t/idx_{i}_recons_{os.path.basename(d['filename'])[:-4]}.png")
    Image.fromarray(img).save(f"t/idx_{i}_img_{os.path.basename(d['filename'])[:-4]}.png")
    Image.fromarray(iframe).save(f"t/idx_{i}_iframe_{os.path.basename(d['filename'])[:-4]}.png")

    mv = clip_and_scale(mv, 20)
    mv += 128
    mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
    mv = Image.fromarray(mv).save(f"t/idx_{i}_mv_{os.path.basename(d['filename'])[:-4]}.png")

    res = (res >> 2) + 128
    res = (np.minimum(np.maximum(res, 0), 255)).astype(np.uint8)
    res = Image.fromarray(res).save(f"t/idx_{i}_res_{os.path.basename(d['filename'])[:-4]}.png")

    # https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html