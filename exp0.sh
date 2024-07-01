#!/bin/bash

ARCHS=(
    'gru'
    'resnet'
    'vgg'
)

MAP="yuv_bitplanes_patches_huffman_fullbit"
W=1
H=2
length=${#ARCHS[@]}
for ((i=0; i<${length}; i++)); do
    EXP="${MAP}_${W}x${H}_${ARCHS[i]}"
    mkdir -p "${EXP}"
    nohup python ldpc.py \
    --dataset cifar10 \
    --map_funcs "${MAP}" \
    --patch_w $W \
    --patch_h $H \
    --arch "${ARCHS[i]}" \
    --fullbit True \
    --output_dir "${EXP}" \
    > "${EXP}/training.log" &
done
