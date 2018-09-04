#!bin/bash

source ../conf/variables.sh

python -u intersect_tumor_region.py ${HEATMAP_TXT_OUTPUT_FOLDER} ${TUMOR_HEATMAPS_PATH} &> ${LOG_OUTPUT_FOLDER}/log.intersect_tumor_region.txt 
wait;
