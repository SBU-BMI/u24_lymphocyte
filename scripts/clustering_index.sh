#!/bin/bash

CANCER_TYPE=seer
OUT_FOL='../cluster_indices'
OUT_DIR='../cluster_indices/inputs'
IN_DIR='../data'
LOG_OUTPUT_FOLDER='../data/log'
W=5
H=5

if [[ ! -z "${OUT_DIR}/*" ]]; then
    echo "cleaning up... ${OUT_DIR}/*"
    rm -rf ${OUT_DIR}/*
fi

cd ../


### generate grayscale heatmap
if [[ ! -z "data/grayscale_heatmaps/*" ]]; then
    echo "cleaning up... data/grayscale_heatmaps/*"
    rm data/grayscale_heatmaps/*
fi
cd ./download_heatmap/get_grayscale_heatmaps/
bash start.sh
cd ../../
wait;

### run thresholding grayscale heatmap
if [[ ! -z "data/thresholded_heatmaps/*" ]]; then
    echo "cleaning up... data/thresholded_heatmaps/*"
    rm data/thresholded_heatmaps/*
fi
cd ./download_heatmap/threshold_grayscale_heatmaps/
bash start.sh
cd ../../
wait;

### copy heatmap over other folders named "rates-<cancer_type>-all-auto" in data
### and heatmap format is *-automatic.png
cd ./scripts
python -u copy_heatmaps.py ${CANCER_TYPE} ${OUT_DIR} 1 &> ${LOG_OUTPUT_FOLDER}/log.copy_heatmaps.txt
wait;
cd ../

cd ./csv_generation
matlab -nodisplay -singleCompThread -r \
"get_patch_til_svs_file_singlefile_wrap('${CANCER_TYPE}', '${OUT_DIR}', '${IN_DIR}', ${W}, ${H}); exit;" \
</dev/null &>${LOG_OUTPUT_FOLDER}/log.get_patch_til_group_svs_file.txt
cd ../
wait;

cd ./scripts
python -u copy_heatmaps.py ${CANCER_TYPE} ${OUT_FOL} 2 &>> ${LOG_OUTPUT_FOLDER}/log.copy_heatmaps.txt
cd ../
wait;

# run the Rscript
cd ./cluster_indices
bash run_all.sh input_full.csv 3 > ${LOG_OUTPUT_FOLDER}/log.cluster_indices_from_csv_files.txt
wait;

bash collateClusterIdx.sh > ${LOG_OUTPUT_FOLDER}/log.collateClusterIndices.txt

exit 0
