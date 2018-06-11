#!/bin/bash

SLIDES=../svs/

for files in ./patch_coordinates/*.txt; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F '----' '{print $1}'`
    USER=`echo ${files} | awk -F'/' '{print $NF}' | awk -F '----' '{print $2}'`
    FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.svs | head -n 1`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.tif | head -n 1`
    fi
    FULL_SVS_NAME=`echo ${FULL_SVS_PATH} | awk -F'/' '{print $NF}'`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        echo Image ${FULL_SVS_PATH} does not exist
        continue;
    fi

    python draw_patches.py ${files} ${FULL_SVS_PATH}
done

exit 0
