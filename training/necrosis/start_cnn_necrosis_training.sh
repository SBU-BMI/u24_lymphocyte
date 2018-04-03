#!/bin/bash
# start training the CNN for necrosis segmentation
# inputs:
#   Training data under ${CNN_TRAINING_DATA}
# output:
#   Convolutional Neural Network model: ${LYM_NECRO_CNN_MODEL_PATH}/cnn_nec_model.pkl

source ../../conf/variables.sh

THEANO_FLAGS="device=${NEC_CNN_TRAINING_DEVICE}" python -u train_cnn_necrosis.py \
   ${NEC_CNN_TRAINING_DATA} ${LYM_NECRO_CNN_MODEL_PATH} &> ${LOG_OUTPUT_FOLDER}/log.train_cnn_necrosis.txt

exit 0
