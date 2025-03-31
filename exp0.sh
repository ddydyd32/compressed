#!/bin/bash

python ldpc.py --config configs/patch4_gru32_ycifar10.yaml > logs/patch4_gru32_ycifar10.log
python ldpc.py --config configs/patch32_gru32_ycifar10.yaml > logs/patch32_gru32_ycifar10.log

# cp configs/patch32_uresnet18_ycifar10.yaml configs/patch32_uresnet18_ycifar10_huffman.yaml
# cp configs/patch4_uresnet18_ycifar10.yaml configs/patch4_uresnet18_ycifar10_huffman.yaml
# cp configs/patch32_gru32_ycifar10.yaml configs/patch32_gru32_ycifar10_huffman.yaml
# cp configs/patch4_gru32_ycifar10.yaml configs/patch4_gru32_ycifar10_huffman.yaml
# cp configs/patch32_vgg_ycifar10.yaml configs/patch32_vgg_ycifar10_huffman.yaml
# cp configs/patch4_vgg_ycifar10.yaml configs/patch4_vgg_ycifar10_huffman.yaml
