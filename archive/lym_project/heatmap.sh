#!/bin/bash

FILE=$1
LIST=$2

PARAL=$3
MAX_PARAL=$4
GPU=$((PARAL % 2))

LINE_N=0
while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    SVS=`ls -1d svs_tiles/${line}* | awk -F'/' '{print $NF}'`

    THEANO_FLAGS="device=gpu${GPU}" python -u ${FILE}.py svs_tiles/${SVS} > log.${FILE}.${SVS}.txt
done < ${LIST}

exit 0
