#!/bin/bash

cd ../
source ./conf/variables.sh

cd ./download_heatmap/download_markings
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.download_markings.txt
cd ../../

cd ./download_heatmap/get_tumor_labeled_maps
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.get_tumor_labeled_maps.txt
cd ../../

cd ./patch_extraction_from_list
for file in ${TUMOR_IMAGES_TO_EXTRACT}/*.txt; do
    if [ ! -f ${file} ]; then
        continue;
    fi
    bash start.sh ${file} 0 &> ${LOG_OUTPUT_FOLDER}/log.patch_extraction_from_list.txt
done
cd ../

exit 0
