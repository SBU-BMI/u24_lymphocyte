#!/bin/bash

source ./bashrc_theano.sh

FOLDER=$1
PARAL=$2
MAX_PARAL=$3

LINE_N=0
for files in ${FOLDER}/*/; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    if [ -f ${files}/patch-level-color.txt ]; then
        echo ${files}/patch-level-color.txt exists
    else
        echo ${files}/patch-level-color.txt generating
        python color_stats.py ${files}
    fi
done

exit 0
