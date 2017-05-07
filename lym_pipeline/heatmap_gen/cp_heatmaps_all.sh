#!/bin/bash

FOLDER=../patches/
LYM_PRED_VERSION=patch-level-lym.txt
LYM_DIS_FOLDER=./patch-level-lym/

for files in ${FOLDER}/*/${LYM_PRED_VERSION}; do
    dis=`echo ${files} | awk -F'/' '{print "prediction-"substr($(NF-1),1,length($(NF-1))-4);}'`
    cp ${files} ${LYM_DIS_FOLDER}/${dis}
done

exit 0
