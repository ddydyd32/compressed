#!/bin/bash

# nohup python ldpc.py --config_path configs/patch32_uresnet18_ycifar10_huffman.yaml > logs/patch32_uresnet18_ycifar10_huffman.log &
nohup python ldpc.py --config_path configs/patch1x8_uresnet18_ycifar10_huffman.yaml > logs/patch1x8_uresnet18_ycifar10_huffman.log &
# nohup python ldpc.py --config_path configs/patch32_gru32_ycifar10_huffman.yaml > logs/patch32_gru32_ycifar10_huffman.log &
nohup python ldpc.py --config_path configs/patch1x8_gru32_ycifar10_huffman.yaml > logs/patch1x8_gru32_ycifar10_huffman.log &
# nohup python ldpc.py --config_path configs/patch32_vgg_ycifar10_huffman.yaml > logs/patch32_vgg_ycifar10_huffman.log &
nohup python ldpc.py --config_path configs/patch1x8_vgg_ycifar10_huffman.yaml > logs/patch1x8_vgg_ycifar10_huffman.log &

# cp configs/patch32_uresnet18_ycifar10.yaml configs/patch32_uresnet18_ycifar10_huffman.yaml
# cp configs/patch4_uresnet18_ycifar10.yaml configs/patch4_uresnet18_ycifar10_huffman.yaml
# cp configs/patch32_gru32_ycifar10.yaml configs/patch32_gru32_ycifar10_huffman.yaml
# cp configs/patch4_gru32_ycifar10.yaml configs/patch4_gru32_ycifar10_huffman.yaml
# cp configs/patch32_vgg_ycifar10.yaml configs/patch32_vgg_ycifar10_huffman.yaml
# cp configs/patch4_vgg_ycifar10.yaml configs/patch4_vgg_ycifar10_huffman.yaml
