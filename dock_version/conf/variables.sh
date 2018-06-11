#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=quip-data
MONGODB_PORT=27017
FINDAPI_HOST=quip-findapi
FINDAPI_PORT=3000
HEATMAP_VERSION=lym_v1

# Base directory
BASE_DIR=/root/src/lymph_pipeline

# The username you want to download heatmaps from
USERNAME=john.vanarnam@gmail.com
# The list of case_ids you want to download heaetmaps from
CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${BASE_DIR}/data/heatmap_txt
LOG_OUTPUT_FOLDER=${BASE_DIR}/data/log
SVS_INPUT_PATH=${BASE_DIR}/data/svs
PATCH_PATH=${BASE_DIR}/data/patches
PATCH_SAMPLING_LIST_PATH=${BASE_DIR}/data/patch_sample_list
RAW_MARKINGS_PATH=${BASE_DIR}/data/raw_marking_xy
MODIFIED_HEATMAPS_PATH=${BASE_DIR}/data/modified_heatmaps
TUMOR_HEATMAPS_PATH=${BASE_DIR}/data/tumor_labeled_heatmaps
TUMOR_GROUND_TRUTH=${BASE_DIR}/data/tumor_ground_truth_maps
TUMOR_IMAGES_TO_EXTRACT=${BASE_DIR}/data/tumor_images_to_extract
GRAYSCALE_HEATMAPS_PATH=${BASE_DIR}/data/grayscale_heatmaps
THRESHOLDED_HEATMAPS_PATH=${BASE_DIR}/data/thresholded_heatmaps
PATCH_FROM_HEATMAP_PATH=${BASE_DIR}/data/patches_from_heatmap
THRESHOLD_LIST=${BASE_DIR}/data/threshold_list/threshold_list.txt

CAE_TRAINING_DATA=${BASE_DIR}/data/training_data_cae
CAE_TRAINING_DEVICE=gpu0
CAE_MODEL_PATH=${BASE_DIR}/data/models_cae
LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
LYM_CNN_TRAINING_DEVICE=gpu0
LYM_CNN_PRED_DEVICE=gpu0
LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/data/models_cnn
NEC_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
NEC_CNN_TRAINING_DEVICE=gpu1
NEC_CNN_PRED_DEVICE=gpu1

# Load modules
###module purge
###module load matlab
###module load mongodb/3.2.0
###module load jdk8/1.8.0_11
###module load openslide/3.4.0
###module load extlibs/1.0.0
###module load ITK/4.6.1
###module load cuda75
###module load anaconda2/4.4.0
###module load imagemagick/7.0.7
###export PATH=/home/lehhou/git/bin/:${PATH}
###export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/anaconda2/current/lib/"
###export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/cuda75/toolkit/7.5.18/lib64/"
###export CUDA_HOME=/cm/shared/apps/cuda75
###export LIBTIFF_CFLAGS="-I/cm/shared/apps/extlibs/include" 
###export LIBTIFF_LIBS="-L/cm/shared/apps/extlibs/lib -ltiff" 
###source ~/theano/bin/activate

