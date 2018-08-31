#!bin/bash

# inputs: 
  # the prediction-xxx file at high-res located at ${HEATMAP_TXT_OUTPUT_FOLDER} 
  # the labeled map (groundtruth) from download heatmap located at ${TUMOR_HEATMAPS_PATH}
# output: Dice-score and Accuracy printed on the terminal screen or in the logfile ${LOG_OUTPUT_FOLDER}/log.compute_dice_score.txt

# requirement: cv2. Cannot use PIL Image because the labeled heatmaps are very big (~40K X 20K), PIL cannot read such big size images

source ../conf/variables.sh

python -u compute_dice_score.py ${HEATMAP_TXT_OUTPUT_FOLDER} ${TUMOR_HEATMAPS_PATH} &> ${LOG_OUTPUT_FOLDER}/log.compute_dice_score.txt 
wait;
cat ${LOG_OUTPUT_FOLDER}/log.compute_dice_score.txt
