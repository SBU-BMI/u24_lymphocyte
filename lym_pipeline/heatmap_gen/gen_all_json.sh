#!/bin/bash

source ../conf/variables.sh

VER=lym_v8

for files in ./patch-level-merged/prediction-*; do
    SVS=`echo ${files} | awk -F'prediction-|.low_res' '{print $2}'`.svs
    TYPE_N_FOLDER=`bash get_type_n_folder.sh ${SVS}`
    TYPE=`echo ${TYPE_N_FOLDER} | awk '{print $1}'`
    if [ ${TYPE} == "notfound" ]; then
        echo ${SVS} type not found, use default type ${DEFAULT_TYPE}
        TYPE=${DEFAULT_TYPE}
        FOLDER=../svs/
    fi

    if [[ "$files" == *.low_res ]]; then
        python gen_json_multipleheat.py ${files} ${VER}-low_res ${TYPE} ${FOLDER} lym 0.5 necrosis 0.5
    else
        python gen_json_multipleheat.py ${files} ${VER}-high_res ${TYPE} ${FOLDER} lym 0.5 necrosis 0.5
    fi
done

exit 0
