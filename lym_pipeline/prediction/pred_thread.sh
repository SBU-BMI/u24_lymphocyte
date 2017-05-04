#!/bin/bash

source ./bashrc_theano.sh

FOLDER=$1
# PARAL = [0, MAX_PARAL-1]
PARAL=$2
MAX_PARAL=$3
GPU=$((PARAL % 2))

LINE_N=0
for files in ${FOLDER}/*/; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    if [ -f ${files}/patch-level-lym.txt ]; then
        echo ${files}/patch-level-lym.txt exists
    else
        echo ${files}/patch-level-lym.txt generating
        THEANO_FLAGS="device=gpu${GPU}" python pred.py ${files}
    fi
done

exit 0
