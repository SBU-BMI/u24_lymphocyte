#!/bin/bash

for files in /data08/shared/lehhou/necrosis_segmentation_workingdir/svs_info_dir_2/necrosis-prediction*; do
    dis=`echo ${files} | awk -F'necrosis-prediction_' '{print "prediction-"substr($(NF),1,length($(NF))-4);}'`
    echo ${files}
    cp ${files} necrosis/${dis}
done

exit 0
