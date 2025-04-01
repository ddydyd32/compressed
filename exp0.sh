#!/bin/bash

mkdir -p logs

dataset="ycifar10"
map_funcs="yuv_ldpc"
num_bitplanes=8
patch_w=2
patch_h=2
model="gru"
output_dir="${model}_${dataset}_${map_funcs}_${num_bitplanes}_${patch_w}x${patch_h}"
python3 ldpc.py \
    --dataset $dataset \
    --map_funcs "${map_funcs}" \
    --patch_w $patch_w \
    --patch_h $patch_h \
    --model "${model}" \
    --fullbit true \
    --output_dir "logs" > "logs/${output_dir}.log"

arch=resnet
arch=gru