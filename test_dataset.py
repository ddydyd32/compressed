from dataset import MPEG4, clip_and_scale, mv_to_hsv, reconstruct
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
from coviar import load as load_mpeg4

import os
os.makedirs('t', exist_ok=True)

filenames = Path('pytorch-coviar/data/hmdb51/mpeg4_videos/brush_hair/').glob('*mp4')
filenames = list(sorted(map(str, filenames)))[1:2]

dataset = MPEG4(filenames, accumulate=True, GOP_SIZE=12)
loader = DataLoader(dataset, shuffle=False)

it = iter(dataset)
i = -1
ratio = []
while 1:
    # if i >= 11:
    #     break
    i += 1
    print(f"\n[DATA {i}]")
    try:
        d = next(it)
    except:
        break
    # print(f"[dataset.py] [idx {i}] [img]")
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
        # print(f"  {k} [{v2['gop_index']}, {v2['gop_pos']}]:", 'min:', v.min(), 'max:', v.max(), 'mean:', v.mean(), 'shape:', v.shape, v.dtype)
        d[k] = v
    iframe = d['iframe'][..., ::-1]
    res = d['res'][..., ::-1]
    img = d['img'][..., ::-1]
    mv = d['mv']
    nz = (np.abs(mv).sum(-1) != 0).sum()
    print('non zero mv:', nz, mv.shape[0] * mv.shape[1])
    ratio.append(1 - nz / mv.shape[0] / mv.shape[1])
    nz = (np.abs(res).sum(-1) != 0).sum()
    print('non zero res:', nz, mv.shape[0] * mv.shape[1])
    hsv = mv_to_hsv(mv.astype(np.float32))
    Image.fromarray(hsv).save(f"t/idx_{i}_hsv_{os.path.basename(d['filename'])[:-4]}.png")

    recons, move = reconstruct(iframe, mv, res)
    # print('    move:', 'min:', move.min(), 'max:', move.max(), 'mean:', move.mean(), move.dtype, move.shape)
    # print('    recons:', 'min:', recons.min(), 'max:', recons.max(), 'mean:', recons.mean(), recons.dtype, recons.shape)
    # print('    img:', img.mean())
    def proc4save(x):
        return (x * 255).astype(np.uint8)

    # Image.fromarray(proc4save(move)).save(f"t/idx_{i}_move_{os.path.basename(d['filename'])[:-4]}.png")
    # Image.fromarray(proc4save(recons)).save(f"t/idx_{i}_recons_{os.path.basename(d['filename'])[:-4]}.png")
    # Image.fromarray(proc4save(img)).save(f"t/idx_{i}_img_{os.path.basename(d['filename'])[:-4]}.png")
    # Image.fromarray(proc4save(iframe)).save(f"t/idx_{i}_iframe_{os.path.basename(d['filename'])[:-4]}.png")

    mv = clip_and_scale(mv, 20)
    mv += 128
    mv = np.clip(mv, 0, 255)
    # mv = Image.fromarray(mv.astype(np.uint8)).save(f"t/idx_{i}_mv_{os.path.basename(d['filename'])[:-4]}.png")

    # res = res / 2 + 0.5
    # res = 1 - np.clip(res, 0, 1)
    res = 1 - np.abs(res)
    # res = Image.fromarray(proc4save(res)).save(f"t/idx_{i}_res_{os.path.basename(d['filename'])[:-4]}.png")
print('ratio:', np.mean(ratio))