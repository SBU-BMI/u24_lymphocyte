#!/bin/bash

source ../conf/bashrc_theano.sh

FOLDER=$1
# PARAL = [0, MAX_PARAL-1]
PARAL=$2
MAX_PARAL=$3
GPU=$4

LINE_N=0
for files in ${FOLDER}/*/; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    if [ -f ${files}/patch-level-necrosis.txt ]; then
        echo ${files}/patch-level-necrosis.txt exists
    else
        echo ${files}/patch-level-necrosis.txt generating
        THEANO_FLAGS="device=gpu${GPU}" python -u pred_necrosis.py ${files}
    fi
done

exit 0
