#!/bin/bash

LYM_PRED_VERSION=patch-level-pred-4-25-blca_stad_cesc.txt
LIST_FILE=cp_heatmap_list.txt

ALL_LIST_FILE=/tmp/cp_all_heatmaps_list_heatmap_v2.txt
ls -1 /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/*/${LYM_PRED_VERSION} > ${ALL_LIST_FILE}

awk -F'/TCGA-|\\.' 'NR==FNR{k=$0; h[k]=1} NR!=FNR{k="TCGA-"$2; if(k in h){print}}' ${LIST_FILE} ${ALL_LIST_FILE} > ${ALL_LIST_FILE}.tmp
mv ${ALL_LIST_FILE}.tmp ${ALL_LIST_FILE}

while read line; do
    dis=`echo ${line} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    if [ ! -f patch-level/${dis} ]; then
        echo cp ${line} patch-level/${dis}
        cp ${line} patch-level/${dis}
    fi
done < ${ALL_LIST_FILE}

for files in /data08/shared/lehhou/necrosis_segmentation_workingdir/svs_info_dir/necrosis-prediction*; do
    dis=`echo ${files} | awk -F'necrosis-prediction_' '{print "prediction-"substr($(NF),1,length($(NF))-4);}'`
    if [ ! -f necrosis/${dis} ]; then
        echo cp ${files} necrosis/${dis}
        cp ${files} necrosis/${dis}
    fi
done

exit 0
