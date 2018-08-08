#!/bin/bash

cd ../conf
source variables.sh
cd ..

CANCER_TYPE=seer
OUT_DIR=${BASE_DIR}/cluster_indices/inputs
IN_DIR=${BASE_DIR}/data
W=5
H=5
rm -rf ${OUT_DIR}
mkdir ${OUT_DIR}

# generate grayscale heatmap
cd ./download_heatmap/get_grayscale_heatmaps/
bash start.sh
cd ../../
wait;

# run thresholding grayscale heatmap
cd ./download_heatmap/threshold_grayscale_heatmaps/
bash start.sh
cd ../../
wait;

# copy heatmap over other folders named "rates-<cancer_type>-all-auto" in data
# and heatmap format is *-automatic.png
cd ./scripts
python -u copy_heatmaps.py ${CANCER_TYPE} ${OUT_DIR} 1 &> ${LOG_OUTPUT_FOLDER}/log.copy_heatmaps.txt
wait;
cd ..

cd ./csv_generation
matlab -nodisplay -singleCompThread -r \
"get_patch_til_group_svs_file_wrap('${CANCER_TYPE}', '${OUT_DIR}', '${IN_DIR}', ${W}, ${H}); exit;" \
</dev/null &>${LOG_OUTPUT_FOLDER}/log.get_patch_til_group_svs_file.txt
wait;

cd ./scripts
python -u copy_heatmaps.py ${CANCER_TYPE} ${OUT_DIR} 2 &>> ${LOG_OUTPUT_FOLDER}/log.copy_heatmaps.txt
wait;

exit 0
