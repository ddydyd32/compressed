#!/bin/bash

python ldpc.py --config configs/patch4_gru32_ycifar10.yaml > logs/patch4_gru32_ycifar10.log
python ldpc.py --config configs/patch32_gru32_ycifar10.yaml > logs/patch32_gru32_ycifar10.log

MAP="yuv_bitplanes_patches_huffman_fullbit"
W=1
H=8
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
