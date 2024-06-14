import torch
import os

import torchvision
from torchvision.transforms import Normalize
import numpy as np
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import numpy as np
from pyldpc import make_ldpc, ldpc_images
from pyldpc.utils_img import gray2bin, rgb2bin
import evaluate

from transformers import (
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    ViTConfig,
)

from config import config
from model import (
    GRU,
    URESNET18,
    VGG,
    ViT
)


def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    load_auc = evaluate.load("roc_auc", "multiclass")
    load_pre = evaluate.load("precision")
    load_rec = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    prediction_scores = torch.nn.functional.softmax(torch.tensor(logits.astype(np.float32)), -1).detach().cpu().numpy()
    references = labels

    result = {}
    result |= load_accuracy.compute(predictions=predictions, references=references)
    result |= load_f1.compute(predictions=predictions, references=references, average="weighted")
    # print('references:', references.shape, references.max())
    # print('prediction_scores:', prediction_scores.shape)
    # result |= load_auc.compute(prediction_scores=prediction_scores, references=references, multi_class='ovo')
    result |= load_pre.compute(predictions=predictions, references=references, average='micro')
    result |= load_rec.compute(predictions=predictions, references=references, average='micro')
    return result

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

N = config.dataset.patch_w * config.dataset.patch_h
H2, G2 = make_ldpc(N, config.dataset.d_v, config.dataset.d_c, seed=config.dataset.seed, systematic=True, sparse=True)
print('H2:', H2.shape, H2.dtype, 'G2:', G2.shape, G2.dtype)
N1 = 512
H, G = make_ldpc(N1, 4, 8, seed=config.dataset.seed, systematic=True, sparse=True)
print('H:', H.shape, H.dtype, 'G:', G.shape, G.dtype)

def get_patches(x, pw, ph):
    if len(x.shape) == 2:
        x = x[None]
    c, w, h = x.shape
    patches = []
    for i in range(0, w, pw):
        for j in range(0, h, ph):
            patch = x[:, i: i + pw, j: j + ph]
            p = np.zeros((c, pw, ph), dtype=x.dtype)
            p[:, : patch.shape[1], : patch.shape[2]] = patch
            patches.append(p)
    patches = np.stack(patches, 0)
    return patches

def ldpc(x: np.ndarray, pw, ph):
    bitplanes = []
    patches = get_patches(x, pw=pw, ph=ph)
    num_patches, c, _, _ = patches.shape
    bitplanes = np.zeros([config.dataset.num_bitplanes, c, num_patches, pw, ph], dtype=patches.dtype)
    for i in range(config.dataset.num_bitplanes):
        # i = 8 - 1 - i
        mask = 1 << i
        for j, x in enumerate(patches):
            bitplane = (x & mask) >> i
            bitplanes[i, :, j, :, :] = bitplane
    bitplanes = bitplanes.reshape((config.dataset.num_bitplanes * c * num_patches, pw * ph))
    xs = (H2.astype(np.float32) @ bitplanes.T).T
    xs = xs.reshape((config.dataset.num_bitplanes * c, -1))

    return xs

    bitplanes = []
    for i in range(config.dataset.num_bitplanes):
        mask = 1 << i
        bitplane = (x & mask) # >> i
        bin = gray2bin(bitplane) # x: [h, w], bin: [h, w, config.dataset.num_bitplanes]
        coded, noisy = ldpc_images.encode_img(G, bin, config.dataset.snr, seed=config.dataset.seed)
        # decoded = ldpc_images.decode_img(G, H, coded, snrconfig.dataset.snr, bin.shape)
        # assert abs((bitplane) - decoded).mean() == 0
        bitplanes.append(coded)
        # print(f'coded {i}:', coded.mean(), coded.min(), coded.max(), coded.dtype, coded.shape)
    xs = np.zeros([config.dataset.num_bitplanes, N1, max(x.shape[-1] for x in bitplanes)], dtype=x.dtype)
    for i in range(config.dataset.num_bitplanes):
        xs[i, ..., : bitplanes[i].shape[-1]] = bitplanes[i]
    xs = xs.transpose([0, 2, 1]).reshape([xs.shape[0], -1])
    return xs


def yuv_y_patch(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = gray2bin(x).astype(np.float32)
    x = x.transpose([2, 0, 1])[: config.dataset.num_bitplanes]
    patches = get_patches(x, pw=config.dataset.patch_w, ph=config.dataset.patch_h)
    num_patches, c, _, _ = patches.shape
    x = patches.transpose([1, 0, 2, 3]).reshape([c, -1])
    x = torch.tensor(x)
    eg['x'] = x
    return eg


def yuv_y(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = gray2bin(x).astype(np.float32)
    x = x.transpose([2, 0, 1])[: config.dataset.num_bitplanes].reshape(config.dataset.num_bitplanes, -1)
    x = torch.tensor(x)
    eg['x'] = x
    return eg


def yuv_ldpc(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = ldpc(x, pw=config.dataset.patch_w, ph=config.dataset.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.dataset.num_bitplanes, n, t
    eg['x'] = x
    return eg


def yuv_huffman(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = ldpc(x, pw=config.dataset.patch_w, ph=config.dataset.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.dataset.num_bitplanes, n, t
    eg['x'] = x
    return eg


def rgb(eg):
    x = eg['img'].convert('RGB')
    x = np.asarray(x)
    x = rgb2bin(x).astype(np.float32)
    c = 3
    x = x.transpose([2, 0, 1]).reshape([8, c, x.shape[0], x.shape[1]])
    x = x[: config.dataset.num_bitplanes].reshape(config.dataset.num_bitplanes * c, -1)
    x = torch.tensor(x)
    eg['x'] = x
    return eg


def rgb_ldpc(eg):
    x = eg['img'].convert('RGB')
    x = np.asarray(x).transpose([2, 0, 1])
    x = ldpc(x, pw=config.dataset.patch_w, ph=config.dataset.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.dataset.num_bitplanes, n, t
    eg['x'] = x
    return eg


def cifar100(eg):
    eg['label'] = eg['coarse_label']
    return eg


map_funcs = {
    'yuv_y_patch': yuv_y_patch,
    'yuv_y': yuv_y,
    'rgb': rgb,
    'yuv_ldpc': yuv_ldpc,
    'rgb_ldpc': rgb_ldpc,
}


def main():
    data = {
        'test': None,
        'train': None,
    }
    torch.manual_seed(config.dataset.seed)
    np.random.seed(config.dataset.seed)

    dataset_name = config.dataset.name
    for key in list(data.keys()):
        cached = f'data/{dataset_name}_{config.dataset.map_funcs}_pw{config.dataset.patch_w}_ph{config.dataset.patch_h}_{key}'
        try:
            data[key] = load_from_disk(cached)
            print(f'loaded {key} from {cached}')
        except:
            data[key] = load_dataset(f'{dataset_name}')[key]
            # data[key] = data[key].shuffle().select(range(1*config.training.per_device_train_batch_size))
            if config.dataset.map_funcs:
                data[key] = data[key].map(map_funcs[config.dataset.map_funcs], remove_columns=['img'])
            if '100' in dataset_name:
                data[key] = data[key].map(cifar100, remove_columns=['coarse_label', 'fine_label'])
            data[key].set_format(type='torch')
            os.makedirs('data', exist_ok=True)
            data[key].save_to_disk(cached)
        print(key, len(data[key]), data[key])
        data[key + '_loader'] = torch.utils.data.DataLoader(data[key], batch_size=config.training.per_device_train_batch_size, shuffle=key=='train')

    num_classes = 100 if '100' in dataset_name else 10

    # get some random training images
    for batch in data['train']:
        images, labels = batch['x'], batch['label']
        print('image:', images.mean(), images.min(), images.max(), images.dtype, images.shape)
        print('labels:', labels.shape)
        c_in = images.shape[-2]
        break

    stats = {}
    for run in range(config.num_runs):
        torch.manual_seed(config.dataset.seed+run)
        np.random.seed(config.dataset.seed+run)
        if config.model.name == 'gru':
            model = GRU(
                c_in=c_in,
                gru_units=config.model.gru_units,
                num_classes=num_classes
            )
        elif config.model.name == 'resnet':
            model = URESNET18(c_in=c_in, num_classes=num_classes)
        elif config.model.name == 'vgg':
            model = VGG(c_in=c_in, num_classes=num_classes)
        elif config.model.name == 'vit':
            conf = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            conf.hidden_size = c_in
            model = ViT(config=conf, num_classes=num_classes)
        model = model.to('cuda')

        '''
        optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

        for epoch in range(config.training.num_train_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, batch in tqdm(enumerate(data['train_loader'], 0), desc=f'Epoch {epoch}'):
                inputs, labels = batch['x'], batch['label']
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                outputs = model(inputs, labels)
                loss = outputs['loss']
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

        model.eval()
        with torch.no_grad():
            for batch in data['test_loader']:
                inputs, labels = batch['x'], batch['label']
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs['logits'].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                _, predictions = torch.max(outputs['logits'], 1)

        print(f'Accuracy: {100 * correct / total} %')
        # '''
        data_collator = DefaultDataCollator()
        training_args = TrainingArguments(seed=config.dataset.seed+run, data_seed=config.dataset.seed+run, **config.training)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        m = trainer.evaluate(eval_dataset=data['test'])
        print(f'run {run}:', m)
        for k, v in m.items():
            stats[k] = stats.get(k, []) + [v]
        # '''
    print('******** mean ********')
    for k in stats:
        stats[k] = np.array(stats[k]) * 100
        print(f'{k}:', f'{np.mean(stats[k]):.1f} +- {np.std(stats[k]):.1f}')


if __name__ == '__main__':
    main()
