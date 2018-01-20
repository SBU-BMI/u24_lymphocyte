#!/bin/bash

source ./conf/variables.sh

cd ./download_heatmap/download_markings
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.download_markings.txt
cd ../../

cd ./download_heatmap/get_modified_heatmaps
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.get_modified_heatmaps.txt
cd ../../

cd ./patch_extraction_from_list
for file in ${MODIFIED_HEATMAPS_PATH}/*.csv; do
    if [ ! -f ${file} ]; then
        continue;
    fi
    bash start.sh ${file} 1 &> ${LOG_OUTPUT_FOLDER}/log.patch_extraction_from_list.txt
done
cd ../

exit 0
