#!/bin/bash

source ../conf/bashrc_theano.sh

FOLDER=$1
# PARAL = [0, MAX_PARAL-1]
PARAL=$2
MAX_PARAL=$3
GPU=$4

DATA_FILE=patch-level-lym.txt
EXEC_FILE=pred.py

LINE_N=0
for files in ${FOLDER}/*/; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    if [ -f ${files}/${DATA_FILE} ]; then
        echo ${files}/${DATA_FILE} exists
    else
        echo ${files}/${DATA_FILE} generating
        THEANO_FLAGS="device=gpu${GPU}" python -u ${EXEC_FILE} ${files}
    fi
done

exit 0


