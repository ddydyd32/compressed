import torch
import os

import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np
from pyldpc import make_ldpc, ldpc_images
from pyldpc.utils_img import gray2bin

'''
from DCVC_src.models.waseda import (
    Cheng2020Anchor
)
device = 'cuda'
i_frame_load_checkpoint = torch.load('/home/v-dongyaozhu/DCVC/DCVC/checkpoints/cheng2020-anchor-3-e49be189.pth.tar', map_location=torch.device('cpu'))
i_frame_net = Cheng2020Anchor.from_state_dict(i_frame_load_checkpoint).eval()
i_frame_net = i_frame_net.to(device)
i_frame_net.eval()

def get_ldpc(x):
    with torch.no_grad():
        result = i_frame_net(x[None].to(device))
    y = result['y'].cpu()[0]
    z = result['z'].cpu()[0]
    py = result['likelihoods']['y'].cpu()[0]
    pz = result['likelihoods']['z'].cpu()[0]
    return {
        'y': y,
        'z': z,
        'py': py,
        'pz': pz,
    }
'''


B = 8
n=1024
d_v=4
d_c=8
seed=42
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
print('H:', H.shape, H.dtype, 'G:', G.shape, G.dtype)
def ldpc(x: np.ndarray):
    snr = 8
    bitplanes = []
    # for i in range(B):
    #     mask = 1 << i
    #     bitplane = (x & mask) >> i
    #     bitplanes.append(bitplane.flatten())
    # bitplanes = np.stack(bitplanes, 1)
    # xs = (H.astype(np.float32) @ bitplanes).T
    # return xs
    xs = []

    for i in range(B):
        mask = 1 << i
        bitplane = (x & mask) # >> i
        bin = gray2bin(bitplane) # x: [h, w], bin: [h, w, B]
        coded, noisy = ldpc_images.encode_img(G, bin, snr, seed=seed)
        # decoded = ldpc_images.decode_img(G, H, coded, snr, bin.shape)
        # assert abs((bitplane) - decoded).mean() == 0
        # coded = H.astype(np.float32) @ (bitplane.flatten()) # 512,
        xs.append(coded)
        print(f'coded {i}:', coded.mean(), coded.min(), coded.max(), coded.dtype, coded.shape)
    x = np.zeros([B, max(x.shape[-1] for x in xs)], dtype=x.dtype)
    x = np.zeros([B, n, max(x.shape[-1] for x in xs)], dtype=x.dtype)
    for i in range(B):
        x[i, ..., : xs[i].shape[-1]] = xs[i]
    return x


# 1, 1024
def yuv(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]#.astype(np.float32) / 255
    x = gray2bin(x).astype(np.float32)
    x = torch.tensor(x).permute([2, 0, 1]).reshape(B, -1)
    # mean = std = tuple(0.5 for _ in range(x.shape[0]))
    # x = Normalize(mean, std, False)(x)
    eg['x'] = x
    return eg


# 8, 512
def yuv_ldpc(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = ldpc(x).astype(np.float32)# / 255
    x = torch.tensor(x) # b, B, n, t
    # mean = tuple(4 for i in range(x.shape[0]))
    # std = tuple(2 for i in range(x.shape[0]))
    # x = Normalize(mean, std, False)(x[..., None])[..., 0]
    eg['x'] = x
    return eg


batch_size = 64
data = {
    'train': None,
    'test': None,
}
torch.manual_seed(0)
np.random.seed(0)
map_funcs = {
    'yuv': yuv,
    'yuv_ldpc': yuv_ldpc,
}
use_map = 'yuv_ldpc'
# use_map = 'yuv'
for key in data:
    cached = f'data/cifar10_{use_map}_{key}'
    try:
        data[key] = load_from_disk(cached)
        print(f'loaded {key} from {cached}')
    except:
        data[key] = load_dataset('cifar10')[key]
        # data[key] = data[key].shuffle().select(range(1*batch_size))
        if use_map:
            data[key] = data[key].map(map_funcs[use_map], remove_columns=['img'])
        data[key].set_format(type='torch')
        os.makedirs('data', exist_ok=True)
        data[key].save_to_disk(cached)
    print(key, len(data[key]))
    data[key] = torch.utils.data.DataLoader(data[key], batch_size=batch_size, shuffle=key=='train')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
for batch in data['train']:
    images, labels = batch['x'], batch['label']
    print('image:', images.mean(), images.min(), images.max(), images.dtype, images.shape)
    print('labels:', labels.shape)
    c_in = images.shape[1]
    break

class GRU(nn.Module):
    def __init__(self, gru_units=12, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(c_in, gru_units, batch_first=True)
        self.fc = nn.Linear(gru_units, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2) # b t c
        x, h = self.gru(x) # b t c
        x = x[:, -1]
        if self.training:
            x = F.dropout(x, p=0.2)
        x = self.fc(x)
        return x


net = GRU().to('cuda')
# net = torchvision.models.vgg11(num_classes=len(classes)).to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batch in tqdm(enumerate(data['train'], 0), desc=f'Epoch {epoch}'):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch['x'], batch['label']
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f' loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
net.eval()
with torch.no_grad():
    for batch in data['test']:
        inputs, labels = batch['x'], batch['label']
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    if total_pred[classname] == 0:
        continue
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} % out of {total_pred[classname]}')

print(f'Accuracy: {100 * correct / total} %')