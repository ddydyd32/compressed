import torch
from torch.utils.data import Dataset
import numpy as np
from coviar import load as load_mpeg4
from coviar import get_num_gops, get_num_frames


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


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
    def __init__(self, filenames, accumulate=True, GOP_SIZE=12):
        self.accumulate = accumulate
        self.GOP_SIZE = GOP_SIZE
        self.frames = []
        self.types = {
            # 'img': 0,
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
            print('reading...filename:', f, 'num frames:', num_frames)

    def __getitem__(self, idx):
        video = self.frames[idx]
        x = {
            'filename': video['filename'],
            'num_frames': video['num_frames'],
            'gop_index': get_gop_pos(idx, 0, self.GOP_SIZE)[0],
            'gop_pos': get_gop_pos(idx, 0, self.GOP_SIZE)[1],
        }
        for t, i in self.types.items():
            if t in {'mv', 'res'} and video['frame_idx'] % self.GOP_SIZE == 0:
                x[t] = np.zeros(x['iframe']['tensor'].shape, dtype=np.int32)[:, :, :2 if t == 'mv' else 3]
            else:
                # print(f'\n[{t}] load args:', video[t]['gop_index'], video[t]['gop_pos'], i, self.accumulate, video['filename'])
                x[t] = load_mpeg4(video['filename'], video[t]['gop_index'], video[t]['gop_pos'], i, self.accumulate)
            x[t] = {
                'tensor': torch.tensor(x[t]),
                'gop_index': video[t]['gop_index'],
                'gop_pos': video[t]['gop_pos'],
            }
        return x

    def __len__(self):
        return len(self.frames)
