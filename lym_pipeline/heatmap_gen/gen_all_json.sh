#!/bin/bash

source ../conf/variables.sh

FOLDER=../svs/
TYPE=${DEFAULT_TYPE}
VER=lym_v8

for files in ./patch-level-merged/prediction-*; do
    if [[ "$files" == *.low_res ]]; then
        python gen_json_multipleheat.py ${files} ${VER}-low_res ${TYPE} ${FOLDER} lym 0.5 necrosis 0.5
    else
        python gen_json_multipleheat.py ${files} ${VER}-high_res ${TYPE} ${FOLDER} lym 0.5 necrosis 0.5
    fi
done

exit 0
