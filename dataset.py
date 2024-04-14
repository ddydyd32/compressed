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
    if representation in ['residual', 'mv']:
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
        self.videos = [
            {
                'filename': f,
                'num_gops': get_num_gops(f),
                'num_frames': get_num_frames(f),
            } 
            for f in filenames
        ]
        self.IFRAME, self.MV, self.RES = 'iframe', 'mv', 'res'
        for f in filenames:
            num_frames = get_num_frames(f)
            for frame_idx in range(num_frames):
                v = {
                    'filename': f,
                    'num_frames': num_frames,
                    'frame_idx': frame_idx,
                }
                for i, t in enumerate(self.IFRAME, self.MV, self.RES):
                    gop_index, gop_pos = get_gop_pos(frame_idx, i, self.GOP_SIZE)
                    v[t] = {
                        'gop_index': gop_index,
                        'gop_pos': gop_pos
                    }
                self.videos.append(v)
            print('reading...filename:', f, 'num frames:', num_frames)

    def __getitem__(self, idx):
        video = self.videos[idx]
        iframe = load_mpeg4(video['filename'], video[self.IFRAME]['gop_index'], video[self.IFRAME]['frame_idx'], self.accumulate)
        mv = load_mpeg4(video['filename'], video[self.MV]['gop_index'], video[self.MV]['frame_idx'], self.accumulate)
        res = load_mpeg4(video['filename'], video[self.RES]['gop_index'], video[self.RES]['frame_idx'], self.accumulate)
        return {
            'iframe': torch.tensor(iframe),
            'mv': torch.tensor(mv),
            'res': torch.tensor(res),
            'filename': video['filename'],
            'num_frames': video['num_frames'],
        }

    def __len__(self):
        return len(self.videos)
