#!bin/bash

# inputs: 
  # the prediction-xxx files at ${HEATMAP_TXT_OUTPUT_FOLDER} 
  # the labeled map (groundtruth) from download heatmap located at ${TUMOR_HEATMAPS_PATH}
# output: prediction-xxx.intersected files located at ${TUMOR_HEATMAPS_PATH}
# after got the output, run the heatmap_gen code to generate new json files (change the conf/variables.sh file accordingly for new version name)

# requirement: cv2. Cannot use PIL Image because the labeled heatmaps are very big (~40K X 20K), 
# PIL cannot read such big size images


source ../conf/variables.sh

python -u intersect_tumor_region.py ${HEATMAP_TXT_OUTPUT_FOLDER} ${TUMOR_HEATMAPS_PATH} &> ${LOG_OUTPUT_FOLDER}/log.intersect_tumor_region.txt 
wait;
