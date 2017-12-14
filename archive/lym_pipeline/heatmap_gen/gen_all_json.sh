#!/bin/bash

source ../conf/variables.sh

FOLDER=../svs/
VER=lym_v0

for files in ./patch-level-merged/prediction-*; do
    if [[ "$files" == *.low_res ]]; then
        python gen_json_multipleheat.py ${files} ${VER}-low_res ${CANCER_TYPE} ${FOLDER} lym 0.5 necrosis 0.5
    else
        python gen_json_multipleheat.py ${files} ${VER}-high_res ${CANCER_TYPE} ${FOLDER} lym 0.5 necrosis 0.5
    fi
done

exit 0
