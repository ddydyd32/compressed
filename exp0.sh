#!/bin/bash

mkdir -p logs

dataset="cifar10"
MAP_FUNCS=(
    yuv_y
    rgb
)
num_bitplanes=8
dataset="cifar100"
echo "dataset: $dataset"
# for map_funcs in ${MAP_FUNCS[@]} ; do
#     patch_w=32
#     output_dir="${dataset}_${map_funcs}_${num_bitplanes}_${patch_w}x${patch_w}"
#     echo "  map_funcs: $map_funcs"
#     echo "    patch_w: $patch_w"
#     echo "        output_dir: logs/$output_dir.log"
#     python3 ldpc.py \
#         --dataset $dataset \
#         --map_funcs "${map_funcs}" \
#         --patch_w $patch_w \
#         --patch_h $patch_w \
#         --fullbit True \
#         --output_dir "logs" > "logs/${output_dir}.log"
# done
MAP_FUNCS2=(
    rgb_ldpc
)
PATCH_W=(
    4
    # 8
    # 16
    32
)
for map_funcs in ${MAP_FUNCS2[@]} ; do
    echo "  map_funcs: $map_funcs"
    for patch_w in ${PATCH_W[@]} ; do
        echo "    patch_w: $patch_w"
        output_dir="${dataset}_${map_funcs}_${num_bitplanes}_${patch_w}x${patch_w}"
        echo "        output_dir: logs/$output_dir.log"
        python3 ldpc.py \
            --dataset $dataset \
            --map_funcs "${map_funcs}" \
            --patch_w $patch_w \
            --patch_h $patch_w \
            --fullbit True \
            --output_dir "logs" > "logs/${output_dir}.log"
    done
done
