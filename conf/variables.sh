#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=localhost
MONGODB_PORT=27017
HEATMAP_VERSION=lym_v1

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=/data01/shared/lehhou/u24_lymphocyte/data/heatmap_jsons
LOG_OUTPUT_FOLDER=/data01/shared/lehhou/u24_lymphocyte/data/log
SVS_INPUT_PATH=/data01/shared/lehhou/u24_lymphocyte/data/svs
PATCH_PATH=/data01/shared/lehhou/u24_lymphocyte/data/patches

CAE_TRAINING_DATA=/data01/shared/lehhou/u24_lymphocyte/data/training_data_cae
CAE_TRAINING_DEVICE=gpu0
CAE_MODEL_PATH=/data01/shared/lehhou/u24_lymphocyte/data/models_cae
LYM_CNN_TRAINING_DATA=/data01/shared/lehhou/u24_lymphocyte/data/training_data_cnn
LYM_CNN_TRAINING_DEVICE=gpu0
LYM_CNN_PRED_DEVICE=gpu0
LYM_NECRO_CNN_MODEL_PATH=/data01/shared/lehhou/u24_lymphocyte/data/models_cnn
NEC_CNN_TRAINING_DATA=/data01/shared/lehhou/u24_lymphocyte/data/training_data_cnn
NEC_CNN_TRAINING_DEVICE=gpu1
NEC_CNN_PRED_DEVICE=gpu1

# Load modules
module purge
module load matlab/mcr-2014b
module load mongodb/3.2.0
module load jdk8/1.8.0_11
module load openslide/3.4.0
module load extlibs/1.0.0
module load ITK/4.6.1
module load cuda75
module load anaconda2/4.4.0
export PATH=/home/lehhou/git/bin/:${PATH}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/anaconda2/current/lib/"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/cm/shared/apps/cuda75/toolkit/7.5.18/lib64/"
export CUDA_HOME=/cm/shared/apps/cuda75
export LIBTIFF_CFLAGS="-I/cm/shared/apps/extlibs/include" 
export LIBTIFF_LIBS="-L/cm/shared/apps/extlibs/lib -ltiff" 
source ~/theano/bin/activate

