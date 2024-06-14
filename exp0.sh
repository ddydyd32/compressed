#!/bin/bash

# cp configs/nocoding_uresnet18_cifar10rgb.yaml configs/nocoding_uresnet18_cifar100rgb.yaml
# cp configs/patch32_uresnet18_cifar10rgb.yaml configs/patch32_uresnet18_cifar100rgb.yaml
# cp configs/patch4_uresnet18_cifar10rgb.yaml configs/patch4_uresnet18_cifar100rgb.yaml
nohup python ldpc.py --config_path configs/nocoding_uresnet18_cifar100rgb.yaml > logs/nocoding_uresnet18_cifar100rgb.log &
nohup python ldpc.py --config_path configs/patch32_uresnet18_cifar100rgb.yaml > logs/patch32_uresnet18_cifar100rgb.log &
nohup python ldpc.py --config_path configs/patch4_uresnet18_cifar100rgb.yaml > logs/patch4_uresnet18_cifar100rgb.log &

# cp configs/nocoding_vgg_cifar10rgb.yaml configs/nocoding_vgg_cifar100rgb.yaml
# cp configs/patch32_vgg_cifar10rgb.yaml configs/patch32_vgg_cifar100rgb.yaml
# cp configs/patch4_vgg_cifar10rgb.yaml configs/patch4_vgg_cifar100rgb.yaml
nohup python ldpc.py --config_path configs/nocoding_vgg_cifar100rgb.yaml > logs/nocoding_vgg_cifar100rgb.log &
nohup python ldpc.py --config_path configs/patch32_vgg_cifar100rgb.yaml > logs/patch32_vgg_cifar100rgb.log &
nohup python ldpc.py --config_path configs/patch4_vgg_cifar100rgb.yaml > logs/patch4_vgg_cifar100rgb.log &
