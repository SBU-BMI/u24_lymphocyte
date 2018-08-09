#!/bin/bash

cd ../conf
source variables.sh
cd ..

CANCER_TYPE=seer
OUT_DIR=${BASE_DIR}/cluster_indices/inputs
IN_DIR=${BASE_DIR}/data
W=5
H=5
rm -rf ${OUT_DIR}/*   # deleting existing folders/files

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
"get_patch_til_svs_file_singlefile_wrap('${CANCER_TYPE}', '${OUT_DIR}', '${IN_DIR}', ${W}, ${H}); exit;" \
</dev/null &>${LOG_OUTPUT_FOLDER}/log.get_patch_til_group_svs_file.txt
cd ../
wait;

cd ./scripts
python -u copy_heatmaps.py ${CANCER_TYPE} ${OUT_DIR} 2 &>> ${LOG_OUTPUT_FOLDER}/log.copy_heatmaps.txt
cd ../
wait;

# run the Rscript
module load R/4.3.0
cd ./cluster_indices
bash run_all.sh input_full.csv 6 > ${LOG_OUTPUT_FOLDER}/log.generate_clustering_indices_from_csv_files.txt
wait;

bash collateClusterIdx.sh > ${LOG_OUTPUT_FOLDER}/log.collateClusterIndices.txt
wait;

exit 0
