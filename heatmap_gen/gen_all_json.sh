#!/bin/bash

source ../conf/variables.sh

for files in ./patch-level-merged/prediction-*; do
    if [[ "$files" == *.low_res ]]; then
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-low_res  lym 0.5 necrosis 0.5 &>> ../log/log.gen_json_multipleheat.low_res..txt
    else
        python gen_json_multipleheat.py ${files} ${HEATMAP_VERSION}-high_res lym 0.5 necrosis 0.5 &>> ../log/log.gen_json_multipleheat.txt
    fi
done

exit 0
