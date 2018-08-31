#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=xxx.bmi.stonybrook.edu
MONGODB_PORT=27017
HEATMAP_VERSION=lym_v1

# Base directory
BASE_DIR=/data
LOCAL_DIR=/root/u24_lymphocyte

# The username you want to download heatmaps from
USERNAME=ddd
# The list of case_ids you want to download heaetmaps from
CASE_LIST=${LOCAL_DIR}/data/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${BASE_DIR}/results/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${BASE_DIR}/results/heatmap_txt
SVS_INPUT_PATH=${BASE_DIR}/svs
LOG_OUTPUT_FOLDER=${BASE_DIR}/log
LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
PATCH_PATH=${BASE_DIR}/patches

PATCH_SAMPLING_LIST_PATH=${LOCAL_DIR}/data/patch_sample_list
RAW_MARKINGS_PATH=${LOCAL_DIR}/data/raw_marking_xy
MODIFIED_HEATMAPS_PATH=${LOCAL_DIR}/data/modified_heatmaps
TUMOR_HEATMAPS_PATH=${LOCAL_DIR}/data/tumor_labeled_heatmaps
TUMOR_GROUND_TRUTH=${LOCAL_DIR}/data/tumor_ground_truth_maps
TUMOR_IMAGES_TO_EXTRACT=${LOCAL_DIR}/data/tumor_images_to_extract
GRAYSCALE_HEATMAPS_PATH=${LOCAL_DIR}/data/grayscale_heatmaps
THRESHOLDED_HEATMAPS_PATH=${LOCAL_DIR}/data/thresholded_heatmaps
PATCH_FROM_HEATMAP_PATH=${LOCAL_DIR}/data/patches_from_heatmap
THRESHOLD_LIST=${LOCAL_DIR}/data/threshold_list/threshold_list.txt

CAE_TRAINING_DATA=${LOCAL_DIR}/data/training_data_cae
CAE_TRAINING_DEVICE=gpu0
CAE_MODEL_PATH=${BASE_DIR}/models_cae
LYM_CNN_TRAINING_DATA=${BASE_DIR}/training_data_cnn
LYM_CNN_TRAINING_DEVICE=gpu0
LYM_CNN_PRED_DEVICE=gpu0
NEC_CNN_TRAINING_DATA=${LOCAL_DIR}/data/training_data_cnn
NEC_CNN_TRAINING_DEVICE=gpu1
NEC_CNN_PRED_DEVICE=gpu1

