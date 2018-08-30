#!bin/bash

source ../conf/variables.sh

python -u compute_dice_score.py ${HEATMAP_TXT_OUTPUT_FOLDER} ${TUMOR_HEATMAPS_PATH} &> ${LOG_OUTPUT_FOLDER}/log.compute_dice_score.txt 
wait;
cat ${LOG_OUTPUT_FOLDER}/log.compute_dice_score.txt
