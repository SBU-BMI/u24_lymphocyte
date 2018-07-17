#!bin/bash
source ../conf/variables.sh

echo 'Start deleting files in ${PATCH_FROM_HEATMAP_PATH}'
rm -f ${PATCH_FROM_HEATMAP_PATH}/*
echo 'Done delete files in ${PATCH_FROM_HEATMAP_PATH}'

for file in ${TUMOR_IMAGES_TO_EXTRACT}/*.txt; do
    if [ ! -f ${file} ]; then
        continue;
    fi
    echo ${file}
    bash start.sh ${file} 0 &> ${LOG_OUTPUT_FOLDER}/log.patch_extraction_from_list.txt
done

cd ./retrain_data_gen
bash start.sh &> ${LOG_OUTPUT_FOLDER}/log.patct_extract_retrain_data.txt
cd ../

exit 0;
