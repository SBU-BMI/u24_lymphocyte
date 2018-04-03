#!/bin/bash

source ../conf/variables.sh

SLIDES=../svs/
IMG_FOLDER=./annotation_maps/

PRED_FOLDER=../heatmaps_v2/gened/
EMPTY_PRED_HEATMAP=../mix/empty_predictions/

rm -rf patch_coordinates
mkdir patch_coordinates
for files in ${IMG_FOLDER}/*; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'----' '{print $1}'`
    USER=`echo ${files} | awk -F'/' '{print $NF}' | awk -F '----' '{print $2}'`
    FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.svs | head -n 1`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        FULL_SVS_PATH=`ls -1 ${SLIDES}/${SVS}*.tif | head -n 1`
    fi
    FULL_SVS_NAME=`echo ${FULL_SVS_PATH} | awk -F'/' '{print $NF}'`
    if [ ! -f "${FULL_SVS_PATH}" ]; then
        echo image ${FULL_SVS_PATH} does not exist
        continue;
    fi

    PREDICT=`ls -1 ${EMPTY_PRED_HEATMAP}/prediction-${SVS}* | head -n 1`
    if [ ! -f "${PREDICT}" ]; then
        echo Empty prediction ${PREDICT} does not exist
        continue
    fi

    WIDTH=` openslide-show-properties ${FULL_SVS_PATH} | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${FULL_SVS_PATH} | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`
    if [ "${WIDTH}" == "" ]; then
        echo Dimension of image ${FULL_SVS_PATH} is unknown
        continue
    fi

    matlab -nodisplay -singleCompThread -r \
    "get_patch_coordinates('${SVS}', '${USER}', '${IMG}', '${PREDICT}', ${WIDTH}, ${HEIGHT}); exit;" \
    </dev/null
done

exit 0
