#!/bin/bash

source ../../conf/variables.sh

IN_FOLDER=${GRAYSCALE_HEATMAPS_PATH}
OUT_FOLDER=${THRESHOLDED_HEATMAPS_PATH}
THRESHOLD_FILE=${THRESHOLD_LIST}

matlab -nodisplay -singleCompThread -r \
    "auto_thres('${IN_FOLDER}', '${OUT_FOLDER}', '${THRESHOLD_FILE}'); exit;" </dev/null \
    &> ${LOG_OUTPUT_FOLDER}/log.threshold_grayscale_heatmaps.txt

exit 0
