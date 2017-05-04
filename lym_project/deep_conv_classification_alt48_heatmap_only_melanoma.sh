#!/bin/bash

source ~/.bashrc_theano

# PARAL = [0, MAX_PARAL-1]
PARAL=$1
MAX_PARAL=$2
GPU=$((PARAL % 2))

echo staring ${PARAL} ${MAX_PARAL} ${GPU}
cd /data03/shared/lehhou/lym_project/
LINE_N=0
while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    THEANO_FLAGS="device=gpu${GPU}" python -u deep_conv_classification_alt48_heatmap_only_melanoma.py svs_tiles/${line}
done < melanoma_list.txt

exit 0
