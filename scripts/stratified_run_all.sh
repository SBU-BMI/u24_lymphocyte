#!/bin/bash

# keep this file in folder scripts
cd ../
source ./conf/variables.sh

# generate grayscale maps
rm -rf ${GRAYSCALE_HEATMAPS_PATH}/*
cd download_heatmap/get_grayscale_heatmaps/
bash start.sh
cd ../../

wait;


# sample 10 random patches from each slide
# inputs: grayscale maps, pred-xxx files
# outputs: .txt file that contains the locations of patches to be extracted, located in folder patch_samle_list 
rm -rf ${PATCH_SAMPLING_LIST_PATH}/*        # delete all previous .txt file in the folder
cd ./patch_sampling
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.patch_sampling.txt
cd ..

wait;

# extract patches from previous step
# input: .txt file that contains the locations of patches to be extracted, located in folder patch_sample_list
# output: patches in patches_from_heatmap

rm -rf ${PATCH_FROM_HEATMAP_PATH}/*     # delete existing files
cd ./patch_extraction_from_list
for file in ${PATCH_SAMPLING_LIST_PATH}/*.txt; do
    echo ${file}
    if [ ! -f ${file} ]; then
        echo ${file} does not exist
        continue;
    fi
    bash start.sh ${file} 0 &> ${LOG_OUTPUT_FOLDER}/log.patch_extraction_from_list.txt
done
cd ../
wait;

rm -rf ${PATCH_FROM_HEATMAP_PATH}/*original_size*

exit 0
