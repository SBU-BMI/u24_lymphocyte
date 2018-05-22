#!bin/bash

source ../conf/variables.sh

# call functions to generate train/validation data
python -u gen_retrain_data.py  ${PATCH_FROM_HEATMAP_PATH} ${LYM_CNN_TRAINING_DATA}
