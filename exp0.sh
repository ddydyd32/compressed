#!/bin/bash

mkdir -p logs

dataset="cifar10"
MAP_FUNCS=(
    yuv_y
    rgb
    yuv_ldpc
    rgb_ldpc
)
num_bitplanes=8
PATCH_W=(
    4
    8
    32
)
PATCH_H=(
    4
    8
    32
)
models=(
    resnet
    vgg
    gru
)
dataset="cifar10"
echo "dataset: $dataset"
for map_funcs in ${MAP_FUNCS[@]} ; do
    echo "  map_funcs: $map_funcs"
    for patch_w in ${PATCH_W[@]} ; do
        echo "    patch_w: $patch_w"
        for patch_h in ${PATCH_H[@]} ; do
            echo "      patch_h: $patch_h"
            for model in ${models[@]} ; do
                echo "        model: $model"
                output_dir="${model}_${dataset}_${map_funcs}_${num_bitplanes}_${patch_w}x${patch_h}"
                echo "        output_dir: logs/$output_dir.log"
                python3 ldpc.py \
                    --dataset $dataset \
                    --map_funcs "${map_funcs}" \
                    --patch_w $patch_w \
                    --patch_h $patch_h \
                    --model "${model}" \
                    --fullbit True \
                    --output_dir "logs" > "logs/${output_dir}.log"
            done
        done
    done
done
