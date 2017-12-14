#!/bin/bash

LYM_PRED_VERSION=patch-level-pred-3-20-melanoma.txt

while read line; do
    files=/data03/shared/lehhou/lym_project/svs_tiles/${line}*/${LYM_PRED_VERSION}
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    echo cp ${files} patch-level/${dis}
    cp ${files} patch-level/${dis}
done < cp_all_heatmaps.txt

exit 0
