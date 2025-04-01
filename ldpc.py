import torch
import os

import yaml
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
    DataCollatorWithPadding,
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
from entropy import huffman_encode


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


N = config.patch_w * config.patch_h
H2, G2 = make_ldpc(N, config.d_v, config.d_c, seed=config.seed, systematic=True, sparse=True)
print('H2:', H2.shape, H2.dtype, 'G2:', G2.shape, G2.dtype)
N1 = 512
H, G = make_ldpc(N1, 4, 8, seed=config.seed, systematic=True, sparse=True)
print('H:', H.shape, H.dtype, 'G:', G.shape, G.dtype)

def get_patches(x, pw, ph):
    # returns: [num_patches, c, pw, ph]
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
    '''
    bitplanes = np.zeros([config.num_bitplanes, c, num_patches, pw, ph], dtype=patches.dtype)
    for i in range(config.num_bitplanes):
        # i = 8 - 1 - i
        mask = 1 << i
        for j, x in enumerate(patches):
            bitplane = (x & mask) >> i
            bitplanes[i, :, j, :, :] = bitplane
    bitplanes = bitplanes.reshape((config.num_bitplanes * c * num_patches, pw * ph))
    xs = (H2.astype(np.float32) @ bitplanes.T).T
    xs = xs.reshape((config.num_bitplanes * c, -1))

    return xs
    '''

    bitplanes = []
    for i in range(config.num_bitplanes):
        mask = 1 << i
        bitplane = (x & mask) # >> i
        bin = gray2bin(bitplane) # x: [h, w], bin: [h, w, config.num_bitplanes]
        coded, noisy = ldpc_images.encode_img(G, bin, config.snr, seed=config.seed)
        # decoded = ldpc_images.decode_img(G, H, coded, snrconfig.dataset.snr, bin.shape)
        # assert abs((bitplane) - decoded).mean() == 0
        bitplanes.append(coded)
        # print(f'coded {i}:', coded.mean(), coded.min(), coded.max(), coded.dtype, coded.shape)
    xs = np.zeros([config.num_bitplanes, N1, max(x.shape[-1] for x in bitplanes)], dtype=x.dtype)
    for i in range(config.num_bitplanes):
        xs[i, ..., : bitplanes[i].shape[-1]] = bitplanes[i]
    # print('xs:', xs.shape) # xs: (8, 512, 32)
    xs = xs.transpose([0, 2, 1]).reshape([xs.shape[0], -1])
    # print('xs 2:', xs.shape) # 8, 16384
    return xs


def code2np(code, max_len=None):
    if max_len is None:
        max_len = len(code)
    x = np.zeros(max_len).astype(np.float32)
    for j, c in enumerate(code):
        x[-len(code)+j] = int(c)
    return x


def code2fp32(code, scale=False):
    if type(code) is str:
        it = code
    else:
        it = code.reshape(-1)
    x = 0
    for j, c in enumerate(it):
        x = (x << 1) | int(c)
    if scale:
        x = np.array([x - 2 ** 5]).astype(np.float32) / (2 ** 5)
    else:
        x = np.array([x]).astype(np.float32)
    return x


def huffman(x: np.ndarray, pw, ph, fullbit=True):
    bitplanes = []
    patches = get_patches(x, pw=pw, ph=ph) # x: 1 x 32 x 32, pw: 1, ph: 8, 
    num_patches, c, _, _ = patches.shape
    _xs = []
    for i in range(config.num_bitplanes):
        # i = 8 - 1 - i
        mask = 1 << i
        for j, x in enumerate(patches):
            bitplane = (x & mask) >> i
            for k in range(c):
                info = bitplane[k]
                info = code2fp32(info, scale=False) # 0 0 0 0 0 0 1 1 -> 3 for each patch
                _xs.append(info)
    _xs = np.concatenate(_xs).astype(np.int32)
    y = huffman_encode(_xs.reshape(-1), 'prefix', save_dir='./')
    ml = max(len(x) for x in y)
    if fullbit:
        _xs = [code2np(x, max_len=ml) for x in y]
    else:
        _xs = [code2fp32(x, scale=False) for x in y]
    xs = np.concatenate(_xs)
    return xs.reshape(config.num_bitplanes * c, -1)
    # return xs.reshape(config.num_bitplanes * c * num_patches, ml)


def huffman_pix(x: np.ndarray, pw, ph, fullbit=True):
    bitplanes = []
    patches = get_patches(x, pw=pw, ph=ph)
    num_patches, c, _, _ = patches.shape
    _xs = []
    for j, x in enumerate(patches): # todo c first or patch first?
        for k in range(c):
            info = x[k]
            _xs += info.reshape(-1).tolist()
    _xs = np.asarray(_xs).astype(np.int32)
    y = huffman_encode(_xs.reshape(-1), 'prefix', save_dir='./')
    ml = max(len(x) for x in y)
    if fullbit:
        _xs = [code2np(x, max_len=ml) for x in y]
    else:
        _xs = [code2fp32(x, scale=False) for x in y]
    xs = np.concatenate(_xs)
    return xs.reshape(config.num_bitplanes * c, -1)
    # return xs.reshape(config.num_bitplanes * c * num_patches, ml)


def yuv_y(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = gray2bin(x).astype(np.float32)
    x = x.transpose([2, 0, 1])[: config.num_bitplanes].reshape(config.num_bitplanes, -1)
    x = torch.tensor(x)
    eg['x'] = x
    return eg


def yuv_ldpc(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = ldpc(x, pw=config.patch_w, ph=config.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.num_bitplanes, n, t
    eg['x'] = x
    return eg


def yuv_huffman(eg):
    x = eg['img'].convert('YCbCr')
    x = np.asarray(x)[:, :, 0]
    x = huffman(x, pw=config.patch_w, ph=config.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.num_bitplanes, n, t
    eg['x'] = x
    return eg


def rgb(eg):
    x = eg['img'].convert('RGB')
    x = np.asarray(x)
    x = rgb2bin(x).astype(np.float32)
    c = 3
    x = x.transpose([2, 0, 1]).reshape([8, c, x.shape[0], x.shape[1]])
    x = x[: config.num_bitplanes].reshape(config.num_bitplanes * c, -1)
    x = torch.tensor(x)
    eg['x'] = x
    return eg


def rgb_ldpc(eg):
    x = eg['img'].convert('RGB')
    x = np.asarray(x).transpose([2, 0, 1])
    x = ldpc(x, pw=config.patch_w, ph=config.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.num_bitplanes, n, t
    eg['x'] = x
    return eg


def rgb_huffman(eg):
    x = eg['img'].convert('RGB')
    x = np.asarray(x).transpose([2, 0, 1])
    x = huffman_pix(x, pw=config.patch_w, ph=config.patch_h).astype(np.float32)
    x = torch.tensor(x) # b, config.num_bitplanes, n, t
    eg['x'] = x
    return eg


def cifar100(eg):
    eg['label'] = eg['coarse_label']
    eg['label'] = eg['fine_label']
    return eg


map_funcs = {
    'yuv_y': yuv_y,
    'rgb': rgb,
    'yuv_ldpc': yuv_ldpc,
    'rgb_ldpc': rgb_ldpc,
    'yuv_huffman': yuv_huffman,
    'rgb_huffman': rgb_huffman,
}


def main():
    os.makedirs(config.training.output_dir, exist_ok=True)
    # with open(os.path.join(config.training.output_dir, '_config.yaml'), 'w') as file:
    #     yaml.dump(config, file, indent=4)
    os.environ['HF_HOME'] = "C:/Users/cornu/Desktop/compressed/data/.cache"
    data = {
        'test': None,
        'train': None,
    }
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset_name = config.dataset
    for key in list(data.keys()):
        cached = f'data/{dataset_name}_{config.map_funcs}_pw{config.patch_w}_ph{config.patch_h}_{key}_100'
        if os.path.isdir(cached):
            data[key] = load_from_disk(cached)
            print(f'loaded {key} from {cached}')
        else:
            data[key] = load_dataset(f'{dataset_name}', cache_dir="data/.cache", split=key)#, streaming=True)#[key]
            # data[key] = data[key].shuffle().select(range(1*config.training.per_device_train_batch_size))
            if config.map_funcs:
                data[key] = data[key].map(map_funcs[config.map_funcs], remove_columns=['img'], num_proc=os.cpu_count())
            if '100' in dataset_name:
                data[key] = data[key].map(cifar100, remove_columns=['coarse_label', 'fine_label'])
            data[key].set_format(type='pt')
            os.makedirs('data', exist_ok=True)
            data[key].save_to_disk(cached)
        print(key, data[key])
        # data[key + '_loader'] = torch.utils.data.DataLoader(data[key], batch_size=config.training.per_device_train_batch_size, shuffle=key=='train')

    num_classes = 100 if '100' in dataset_name else 10

    # get some random training images
    for batch in data['train']:
        # print(batch)
        images, labels = batch['x'], batch['label']
        print('image:', images.mean(), images.min(), images.max(), images.dtype, images.shape)
        print('labels:', labels)
        c_in = images.shape[-2]
        # c_in = 8
        break
    # exit()

    stats = {}
    for run in range(config.num_runs):
        torch.manual_seed(config.seed+run)
        np.random.seed(config.seed+run)
        if config.model == 'gru':
            model = GRU(
                c_in=c_in,
                gru_units=config.gru_units,
                num_classes=num_classes
            )
        elif config.model == 'resnet':
            model = URESNET18(c_in=c_in, num_classes=num_classes)
        elif config.model == 'vgg':
            model = VGG(c_in=c_in, num_classes=num_classes)
        elif config.model == 'vit':
            conf = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            conf.hidden_size = c_in
            model = ViT(config=conf, num_classes=num_classes)
        model = model.to('cuda')

        data_collator = DefaultDataCollator()
        if 'huffman' in config.map_funcs:
            class Collator(DefaultDataCollator):
                def __call__(self, features):
                    fs = []
                    xs = np.zeros([len(features), features[0]['x'].shape[0], max(z['x'].shape[1] for z in features)]).astype(np.float32)
                    for i, b in enumerate(features):
                        xs[i, :, : b['x'].shape[1]] = b['x']
                        b.pop('x')
                        fs.append(b)
                    output = super().__call__(fs)
                    output['x'] = torch.tensor(xs, device=output['labels'].device)
                    return output
            data_collator = Collator()
        else:
            def collate(features):
                # xs = np.zeros([len(features), 8, 16384]).astype(np.float32)
                # fs = np.ones(len(features))
                fs = []
                xs = []
                for i, b in enumerate(features):
                    xs.append(map_funcs[config.map_funcs](b)['x'])
                    fs.append(b['label'])
                xs = torch.stack(xs, 0)
                ys = torch.tensor(fs).long()
                output = {
                    'x': xs,
                    'labels': ys
                }
                return output
            # data_collator = collate
        training_args = TrainingArguments(
            remove_unused_columns=False,
            seed=config.seed+run,
            data_seed=config.seed+run,
            **config.training
        )

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
        stats[k] = np.array(stats[k])
        print(f'{k}:', f'{np.mean(stats[k]):.1f} +- {np.std(stats[k]):.1f}')


if __name__ == '__main__':
    main()
