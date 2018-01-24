#!/bin/bash

cd ../
cd ./patch_sampling
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.patch_sampling.txt
cd ..

cd ./patch_extraction_from_list
for file in ${TUMOR_IMAGES_TO_EXTRACT}/*.txt; do
    if [ ! -f ${file} ]; then
        continue;
    fi
    bash start.sh ${file} 0 &> ${LOG_OUTPUT_FOLDER}/log.patch_extraction_from_list.txt
done
cd ../

exit 0
