#!/bin/bash

PARAL=$1
MAX_PARAL=$2
GPU=$((PARAL % 2))

LINE_N=0
while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    THEANO_FLAGS="device=gpu${GPU}" python -u deep_conv_classification_alt36_heatmap.py svs_tiles/${line} \
    > log.deep_conv_classification_alt36_heatmap.${line}.txt
done < deep_conv_classification_alt36_heatmap.txt

exit 0
