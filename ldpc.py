import torch
import os

from omegaconf import OmegaConf
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


try:
    N = config.dataset.patch_w * config.dataset.patch_h
    H2, G2 = make_ldpc(N, config.dataset.d_v, config.dataset.d_c, seed=config.dataset.seed, systematic=True, sparse=True)
    print('H2:', H2.shape, H2.dtype, 'G2:', G2.shape, G2.dtype)
    N1 = 512
    H, G = make_ldpc(N1, 4, 8, seed=config.dataset.seed, systematic=True, sparse=True)
    print('H:', H.shape, H.dtype, 'G:', G.shape, G.dtype)
except:
    print('NO LDPC AVAILABLE')

def get_patches(x, pw, ph):
    # returns: [num_patches, c, pw, ph]
    if len(x.shape) == 2:
        x = x[None]
    c, w, h = x.shape
    patches = []
    pw, ph = config.dataset.patch_w, config.dataset.patch_h
    for i in range(0, w, pw):
        for j in range(0, h, ph):
            patch = x[:, i: i + pw, j: j + ph]
            p = np.zeros((c, pw, ph), dtype=x.dtype)
            p[:, : patch.shape[1], : patch.shape[2]] = patch
            patches.append(p)
    # [num_patches, c, pw, ph]
    patches = np.stack(patches, 0)
    return patches


def ldpc(x: np.ndarray):
    bitplanes = []
    pw, ph = config.dataset.patch_w, config.dataset.patch_h
    patches = get_patches(x, pw=pw, ph=ph)
    num_patches, c, _, _ = patches.shape
    '''
    bitplanes = np.zeros([config.datasetset.num_bitplanes, c, num_patches, pw, ph], dtype=patches.dtype)
    for i in range(config.datasetset.num_bitplanes):
        # i = 8 - 1 - i
        mask = 1 << i
        for j, x in enumerate(patches):
            bitplane = (x & mask) >> i
            bitplanes[i, :, j, :, :] = bitplane
    bitplanes = bitplanes.reshape((config.dataset.num_bitplanes * c * num_patches, pw * ph))
    xs = (H2.astype(np.float32) @ bitplanes.T).T
    xs = xs.reshape((config.dataset.num_bitplanes * c, -1))

    return xs
    '''

    bitplanes = []
    for i in range(config.datasetset.num_bitplanes):
        mask = 1 << i
        bitplane = (x & mask) # >> i
        bin = gray2bin(bitplane) # x: [h, w], bin: [h, w, config.datasetset.num_bitplanes]
        coded, noisy = ldpc_images.encode_img(G, bin, config.datasetset.snr, seed=config.datasetset.seed)
        # decoded = ldpc_images.decode_img(G, H, coded, snrconfig.datasetset.snr, bin.shape)
        # assert abs((bitplane) - decoded).mean() == 0
        bitplanes.append(coded)
        # print(f'coded {i}:', coded.mean(), coded.min(), coded.max(), coded.dtype, coded.shape)
    xs = np.zeros([config.datasetset.num_bitplanes, N1, max(x.shape[-1] for x in bitplanes)], dtype=x.dtype)
    for i in range(config.datasetset.num_bitplanes):
        xs[i, ..., : bitplanes[i].shape[-1]] = bitplanes[i]
    print('xs:', xs.shape)
    xs = xs.transpose([0, 2, 1]).reshape([xs.shape[0], -1])
    print('xs 2:', xs.shape)
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
        if it.shape[0] == 1:
            return code.reshape(1)
    x = 0
    for j, c in enumerate(it):
        x = (x << 1) | int(c)
    if scale:
        x = np.array([x - 2 ** 5]).astype(np.float32) / (2 ** 5)
    else:
        x = np.array([x]).astype(np.float32)
    return x


def get_huffman(x: np.ndarray):
    # x: [c, w, h] or [np, c, pw, ph]
    if len(x.shape) == 3:
        x = x[None]
    num_patches, c, _, _ = x.shape
    _xs = []
    bitplanes = []
    x = x.astype(np.int32)
    # print('[get_huffman] x:', x.shape)
    # for i in range(config.dataset.num_bitplanes):
    for i in range(1):
        # i = 8 - 1 - i
        mask = 1 << i
        for j, p in enumerate(x):
            bitplane = (p & mask) >> i
            for k in range(c):
                info = bitplane[k]
                info = code2fp32(info, scale=False) # 0 0 0 0 0 0 1 1 -> 3 for each patch
                # if i == 0 and j == 0 and k == 0:
                    # print('info in:', bitplane[k].shape, bitplane[k], 'info out:', info.shape, info)
                _xs.append(info)
    _xs = np.concatenate(_xs).astype(np.int32)
    # print('_xs:', _xs.shape)
    y = huffman_encode(_xs.reshape(-1), 'prefix', save_dir='./')
    return y


ML = -1
def get_fullbit(y):
    ml = max(len(x) for x in y)
    global ML
    ML = max(ML, ml)
    _xs = [code2np(x, max_len=ml) for x in y]
    xs = np.concatenate(_xs)
    return xs.reshape(-1, ml) # config.dataset.num_bitplanes * c


def get_shortbit(y):
    _xs = [code2fp32(x, scale=False) for x in y]
    xs = np.concatenate(_xs)
    print('xs:', xs.shape)
    return xs.reshape(-1, 1) # config.dataset.num_bitplanes * c, -1


def get_yuv(x):
    x = x.convert('YCbCr')
    x = np.asarray(x)[None, :, :, 0] # [1, w, h]
    return x


def get_rgb(x):
    x = x.convert('RGB')
    x = np.asarray(x).transpose([2, 0, 1]) # [3, w, h]
    return x


def get_bitplanes(x):
    c, w, h = x.shape
    if c == 1:
        x = gray2bin(x[0]).astype(np.float32)
        x = x.transpose([2, 0, 1])[: config.dataset.num_bitplanes]
    elif c == 3:
        x = rgb2bin(x.transpose([1, 2, 0])).astype(np.float32)
        x = x.transpose([2, 0, 1]).reshape([8, c, x.shape[0], x.shape[1]])
        x = x[: config.dataset.num_bitplanes].reshape(config.dataset.num_bitplanes * c, -1)
    return x


def cifar100(eg):
    eg['label'] = eg['coarse_label']
    eg['label'] = eg['fine_label']
    return eg


map_funcs = {
    'yuv': get_yuv,
    'rgb': get_rgb,
    'bitplanes': get_bitplanes,
    'patches': get_patches,
    'huffman': get_huffman,
    'fullbit': get_fullbit,
    'shortbit': get_shortbit,
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
    torch.manual_seed(config.dataset.seed)
    np.random.seed(config.dataset.seed)

    dataset_name = config.dataset.name
    for key in list(data.keys()):
        cached = f'data/{dataset_name}_{config.dataset.map_funcs}_pw{config.dataset.patch_w}_ph{config.dataset.patch_h}_{key}_100'
        try:
            # raise NotImplementedError
            data[key] = load_from_disk(cached)
            print(f'loaded {key} from {cached}')
        except:
            data[key] = load_dataset(f'{dataset_name}', cache_dir="data/.cache")[key]
            # data[key] = data[key].shuffle().select(range(1*config.training.per_device_train_batch_size))
            if config.dataset.map_funcs:
                def f(eg):
                    x = eg['img']
                    for f in config.dataset.map_funcs.split('_'):
                        x = map_funcs[f](x)
                    eg['x'] = x.transpose(1, 0)
                    return eg
                data[key] = data[key].map(f, remove_columns=['img'])
            if '100' in dataset_name:
                data[key] = data[key].map(cifar100, remove_columns=['coarse_label', 'fine_label'])
            data[key].set_format(type='torch')
            # os.makedirs('data', exist_ok=True)
            # data[key].save_to_disk(cached)
        print(key, len(data[key]), data[key])
        data[key + '_loader'] = torch.utils.data.DataLoader(data[key], batch_size=config.training.per_device_train_batch_size, shuffle=key=='train')

    num_classes = 100 if '100' in dataset_name else 10

    # get some random training images
    for batch in data['train']:
        images, labels = batch['x'], batch['label']
        print('image:', images.mean(), images.min(), images.max(), images.dtype, images.shape)
        print('labels:', labels.shape)
        c_in, t = images.shape
        if ML > -1:
            c_in = ML
        print('c_in:', c_in)
        break
    # exit()

    stats = {}
    for run in range(config.experiments.num_runs):
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

        data_collator = DefaultDataCollator()
        if 'huffman' in config.dataset.map_funcs:
            class Collator(DefaultDataCollator):
                def __call__(self, features):
                    fs = []
                    xs = np.zeros([len(features), max(z['x'].shape[0] for z in features), features[0]['x'].shape[1]]).astype(np.float32)
                    for i, b in enumerate(features):
                        xs[i, : b['x'].shape[0], :] = b['x']
                        b.pop('x')
                        fs.append(b)
                    output = super().__call__(fs)
                    output['x'] = torch.tensor(xs, device=output['labels'].device)
                    return output

            data_collator = Collator()
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
        stats[k] = np.array(stats[k])
        print(f'{k}:', f'{np.mean(stats[k]):.1f} +- {np.std(stats[k]):.1f}')


if __name__ == '__main__':
    main()
