#!/bin/bash

# PARAL = [0, MAX_PARAL-1]
PARAL=$1
MAX_PARAL=$2
GPU=$((PARAL % 2))

LINE_N=0
while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi
    SVS=`ls -1d svs_tiles/${line}* | awk -F'/' '{print $NF}'`

    THEANO_FLAGS="device=gpu${GPU}" python -u deep_conv_classification_alt51_heatmap.py svs_tiles/${SVS} \
    > log.deep_conv_classification_alt51_heatmap.${SVS}.txt
done < heatmap_slide_list51.txt

exit 0
