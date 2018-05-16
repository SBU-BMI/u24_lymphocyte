#!/bin/bash

source ../conf/variables.sh

COD_PARA=$1
MAX_PARA=$2
IN_FOLDER=${SVS_INPUT_PATH}
OUT_FOLDER=${PATCH_PATH}

LINE_N=0
for files in ${IN_FOLDER}/*.*; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARA)) -ne ${COD_PARA} ]; then continue; fi

    SVS=`echo ${files} | awk -F'/' '{print $NF}'`
    python save_svs_to_tiles.py $SVS $IN_FOLDER $OUT_FOLDER
    if [ $? -ne 0 ]; then
        echo "failed extracting patches for " ${SVS}
        rm -rf ${OUT_FOLDER}/${SVS}
    fi
done

exit 0;

