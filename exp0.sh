#!/bin/bash

nohup python ldpc.py --config_path configs/patch32_uresnet18_cifar10rgb.yaml > patch32_uresnet18_cifar10rgb.log &
nohup python ldpc.py --config_path configs/patch32_uresnet18_ycifar10.yaml > patch32_uresnet18_ycifar10.log &
nohup python ldpc.py --config_path configs/patch4_uresnet18_cifar10rgb.yaml > patch4_uresnet18_cifar10rgb.log &
nohup python ldpc.py --config_path configs/patch4_uresnet18_ycifar10.yaml > patch4_uresnet18_ycifar10.log &
# nohup python ldpc.py --config_path configs/patch32_gru32_cifar10rgb.yaml > patch32_gru32_cifar10rgb.log &
# nohup python ldpc.py --config_path configs/patch32_gru32_ycifar10.yaml > patch32_gru32_ycifar10.log &
# nohup python ldpc.py --config_path configs/patch4_gru32_cifar10rgb.yaml > patch4_gru32_cifar10rgb.log &
# nohup python ldpc.py --config_path configs/patch4_gru32_ycifar10.yaml > patch4_gru32_ycifar10.log &
