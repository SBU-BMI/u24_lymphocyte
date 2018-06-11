#!/bin/bash

TYPE=read
SAMPLE_RATE=1
LYM_PRED_VERSION=patch-level-pred-4-1-adeno.txt

N=0
for files in /data03/shared/lehhou/lym_project/svs_tiles/*/${LYM_PRED_VERSION}; do
    svs=`echo ${files} | awk -F'/' '{print $(NF-1);}'`
    if [ -f "${TYPE}/${svs}" ]; then
        dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
        if [ ! -f patch-level/${dis} ]; then
            if [ $((N % SAMPLE_RATE)) -eq 0 ]; then
                echo ${files}
                cp ${files} patch-level/${dis}
            fi
            N=$((N+1))
        fi
    fi
done

for files in /data08/shared/lehhou/necrosis_segmentation_workingdir/svs_info_dir/necrosis-prediction*; do
    dis=`echo ${files} | awk -F'necrosis-prediction_' '{print "prediction-"substr($(NF),1,length($(NF))-4);}'`
    if [ ! -f necrosis/${dis} ]; then
        cp ${files} necrosis/${dis}
    fi
done

exit 0
