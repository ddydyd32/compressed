import torch
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from coviar import load as load_mpeg4
from coviar import get_num_gops, get_num_frames


def get_gop_pos(frame_idx, representation, GOP_SIZE):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation == 'img':
        return gop_index, gop_pos
    if representation in ['residual', 'mv', 'res']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class MPEG4(Dataset):
    def __init__(self, filenames, accumulate=True, GOP_SIZE=12, vit_patchsize=14):
        self.accumulate = accumulate
        self.GOP_SIZE = GOP_SIZE
        self.frames = []
        self.types = {
            'img': 0,
            'iframe': 0,
            'mv': 1,
            'res': 2
        }
        for f in filenames:
            num_frames = get_num_frames(f)
            for frame_idx in range(num_frames):
                v = {
                    'filename': f,
                    'num_frames': num_frames,
                    'frame_idx': frame_idx,
                }
                for t, i in self.types.items():
                    gop_index, gop_pos = get_gop_pos(frame_idx, t, self.GOP_SIZE)
                    v[t] = {
                        'gop_index': gop_index,
                        'gop_pos': gop_pos,
                        'type': i
                    }
                self.frames.append(v)
            print('\nreading...filename:', f, 'num frames:', num_frames)
        self.set_vit_patchsize(p)

    def set_vit_patchsize(self, p):
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=p, stride=p, bias=False)
        with torch.no_grad():
            self.conv.weight.data = torch.ones_like(self.conv.weight.data)
        self.vit_patchsize = p

    def __getitem__(self, idx):
        video = self.frames[idx]
        x = {
            'filename': video['filename'],
            'num_frames': video['num_frames'],
            'gop_index': get_gop_pos(idx, 'img', self.GOP_SIZE)[0],
            'gop_pos': get_gop_pos(idx, 'img', self.GOP_SIZE)[1],
        }
        for t, i in self.types.items():
            if t in {'mv', 'res'} and video['frame_idx'] % self.GOP_SIZE == 0:
                x[t] = np.zeros(x['iframe']['tensor'].shape, dtype=np.int32)[:, :, :2 if t == 'mv' else 3]
            else:
                x[t] = load_mpeg4(video['filename'], video[t]['gop_index'], video[t]['gop_pos'], i, self.accumulate)
            if t in ['img', 'iframe', 'res']:
                x[t] = x[t].astype(np.float32) / 255
            x[t] = {
                'tensor': torch.tensor(x[t]),
                'gop_index': video[t]['gop_index'],
                'gop_pos': video[t]['gop_pos'],
            }
        W, H, _ = mask.shape
        PATCH_SIZE = self.vit_patchsize
        mask = mask.abs().reshape([W // PATCH_SIZE, PATCH_SIZE, H // PATCH_SIZE, PATCH_SIZE, -1])
        mask = mask.transpose(2, 1).sum([-1, -2, -3])
        # conv = torch.nn.Conv2d(3, 1, kernel_size=vit_patchsize, stride=vit_patchsize, bias=False)
        # with torch.no_grad():
        #     conv.weight.data = torch.ones_like(conv.weight.data)
        # with torch.no_grad():
        #     mask = mask.abs().permute(2, 0, 1)
        #     mask = conv(mask).squeeze(0)
        x['vit_patch_mask_1_to_keep'] = mask > 0
        return x

    def __len__(self):
        return len(self.frames)


# img: [-size, size]
def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


# https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
def mv_to_hsv(mv):
    # mv: [x, y, [dy, dx]]
    magnitude, angle = cv2.cartToPolar(mv[:, :, 1], mv[:, :, 0], angleInDegrees=True)
    hsv = np.zeros([mv.shape[0], mv.shape[1], 3], dtype=mv.dtype)
    hsv[:, :, 0] = angle * 180. / 360 / 255
    hsv[:, :, 1] = 1.
    hsv[:, :, 2] = cv2.normalize(magnitude, magnitude, 0, 1, cv2.NORM_MINMAX)
    hsv = (hsv * 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return hsv


def reconstruct(ref, mv, res):
    if np.abs(mv).sum() == 0:
        return ref, ref
    height, width, _ = ref.shape
    yv, xv = np.meshgrid(range(width), range(height))
    dx, dy = mv[:, :, 1], mv[:, :, 0]
    src_x, src_y = xv - dx, yv - dy
    src_x = np.clip(src_x, 0, height-1)
    src_y = np.clip(src_y, 0, width-1)
    move = ref[src_x, src_y]
    move[src_x * (dx == 0), src_y * (dy == 0)] = 0
    recons = move + res
    recons = np.clip(recons, 0, 1)
    move = np.clip(move, 0, 1)
    return recons, move


def get_patch_index(x, y, patch_size=14, N=224, flatten=False):
    i, j = x // patch_size, y // patch_size
    if flatten:
        return i * (N // patch_size) + j
    return i, j


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from pathlib import Path

    filenames = Path('pytorch-coviar/data/hmdb51/mpeg4_videos/brush_hair/').glob('*mp4')
    filenames = list(sorted(map(str, filenames)))[:1]

    dataset = MPEG4(filenames, accumulate=True, GOP_SIZE=12)
    loader = DataLoader(dataset, shuffle=False)

    it = iter(dataset)
    d = next(it)
    for k, v in d.items():
        if type(v) in [str, int]:
            continue
        v2 = v
        v = v['tensor'].cpu().numpy()
        print(f"  {k} [{v2['gop_index']}, {v2['gop_pos']}]:", 'min:', v.min(), 'max:', v.max(), 'mean:', v.mean(), 'shape:', v.shape, v.dtype)
        d[k] = v
