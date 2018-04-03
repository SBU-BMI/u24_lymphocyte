#!/bin/bash

SAMPLE=1
LYM_PRED_VERSION=patch-level-pred-3-21-paad.txt

N=0
for files in /data03/shared/lehhou/lym_project/svs_tiles/*/${LYM_PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    if [ ! -f patch-level/${dis} ]; then
        if [ $((N % SAMPLE)) -eq 0 ]; then
            cp ${files} patch-level/${dis}
        fi
        N=$((N+1))
    fi
done

for files in /data08/shared/lehhou/necrosis_segmentation_workingdir/svs_info_dir/necrosis-prediction*; do
    dis=`echo ${files} | awk -F'necrosis-prediction_' '{print "prediction-"substr($(NF),1,length($(NF))-4);}'`
    if [ ! -f necrosis/${dis} ]; then
        cp ${files} necrosis/${dis}
    fi
done

exit 0
