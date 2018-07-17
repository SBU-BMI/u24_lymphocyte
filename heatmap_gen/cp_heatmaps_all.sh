#!/bin/bash

FOLDER=${1}     #/data/patches

PRED_VERSION=patch-level-lym.txt
DIS_FOLDER=./patch-level-lym/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
    ### python code to remove prediction of 0
#    python -u prediction_filter.py ${DIS_FOLDER}/${dis} ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-necrosis.txt
DIS_FOLDER=./patch-level-nec/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
done

PRED_VERSION=patch-level-color.txt
DIS_FOLDER=./patch-level-color/
for files in ${FOLDER}/*/${PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "color-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${DIS_FOLDER}/${dis}
done

exit 0
