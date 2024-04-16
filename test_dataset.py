from dataset import MPEG4, clip_and_scale
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
from coviar import load as load_mpeg4

import os
os.makedirs('t', exist_ok=True)

filenames = Path('pytorch-coviar/data/hmdb51/mpeg4_videos/brush_hair/').glob('*mp4')
filenames = list(sorted(map(str, filenames)))[:1]

dataset = MPEG4(filenames, accumulate=True, GOP_SIZE=12)
loader = DataLoader(dataset, shuffle=False)

it = iter(dataset)
i = -1
while 1:
    if i >= 11:
        break
    i += 1
    print(f"\n[DATA {i}]")
    d = next(it)
    print(f"[dataset.py] [idx {i}] [img]")
    # d['img'] = {
    #     'tensor': torch.tensor(load_mpeg4(d['filename'], d['gop_index'], d['gop_pos'], 0, True)),
    #     'gop_index': d['gop_index'],
    #     'gop_pos': d['gop_pos'],
    # }
    for k, v in d.items():
        if type(v) in [str, int]:
            continue
        v2 = v
        v = v['tensor'].cpu().numpy()
        print(f"  {k} [{v2['gop_index']}, {v2['gop_pos']}]:", 'min:', v.min(), 'max:', v.max(), 'mean:', v.mean(), 'shape:', v.shape, v.dtype)
        d[k] = v
    iframe = d['iframe'][..., ::-1]
    res = d['res'][..., ::-1]
    img = d['img'][..., ::-1]
    mv = d['mv']
    height, width, _ = img.shape

    yv, xv = np.meshgrid(range(width), range(height))
    dx, dy = mv[:, :, 1], mv[:, :, 0]
    if i % 12 == 0:
        assert np.abs(iframe[xv, yv] - img).sum() < 1e-5
        assert np.abs(res).sum() < 1e-5
        assert np.abs(mv).sum() < 1e-5
    src_x, src_y = xv - dx, yv - dy
    src_x = np.clip(src_x, 0, height-1)
    src_y = np.clip(src_y, 0, width-1)
    move = iframe[src_x, src_y].astype(np.float32)
    print('    move:', 'min:', move.min(), 'max:', move.max(), 'mean:', move.mean(), move.dtype, move.shape)
    recons = move + res.astype(np.float32)
    print('    recons:', 'min:', recons.min(), 'max:', recons.max(), 'mean:', recons.mean(), recons.dtype, recons.shape)
    print('    img:', img.mean())
    diff = recons - img
    f = np.where(diff != 0)
    if len(f[0]) > 0:
        fx, fy, _ = f
        print('diff:', fx.shape, fx.max(), fy.shape, fy.max(), fx[:15:3], fy[:15:3])
        print(np.vstack([fx, fy])[:, :15:3], np.vstack([fx, fy])[:, -15::3])
        print('res [0, 112]', res[0, 112], '[0, 113]', res[0, 113])
        print('img [0, 112]', img[0, 112], '[0, 113]', img[0, 113])
        print('mv [0, 112]', mv[0, 112], '[0, 113]', mv[0, 113])
        print('src_x [0, 112]', src_x[0, 112], '[0, 113]', src_x[0, 113])
        print('src_y [0, 112]', src_y[0, 112], '[0, 113]', src_y[0, 113])
        print('move [0, 112]', move[0, 112], '[0, 113]', move[0, 113])
        print('recons [0, 112]', recons[0, 112], '[0, 113]', recons[0, 113])
        recons[0, :100, 0] += 255
        print('diff [0, 112]', diff[0, 112], '[0, 113]', diff[0, 113])
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