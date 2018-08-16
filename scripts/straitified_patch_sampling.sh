#!/bin/bash

cd ../
source ./conf/variables.sh

cd ./patch_sampling
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.patch_sampling.txt

cd ..

cd ./patch_extraction_from_list
for file in ${PATCH_SAMPLING_LIST_PATH}/*.txt; do
    if [ ! -f ${file} ]; then
        continue;
    fi
    bash start.sh ${file} 0 &> ${LOG_OUTPUT_FOLDER}/log.patch_extraction_from_list.txt
done
cd ../

exit 0
