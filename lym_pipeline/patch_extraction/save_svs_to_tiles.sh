#!/bin/bash

COD_PARA=$1
MAX_PARA=$2
IN_FOLDER=../svs/
OUT_FOLDER=../patches/

LINE_N=0
for files in ${IN_FOLDER}/*; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARA)) -ne ${COD_PARA} ]; then continue; fi

    SVS=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -d ${OUT_FOLDER}/${SVS} ]; then
        echo $SVS sh existed
    else
        echo $SVS sh extracting
        python save_svs_to_tiles.py $SVS $IN_FOLDER $OUT_FOLDER
    fi
done

exit 0;

