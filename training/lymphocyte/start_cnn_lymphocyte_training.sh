#!/bin/bash
# start training the CNN
# inputs:
#   Convolutional Autoencoder model: ${CAE_MODEL_PATH}/cae_model.pkl
#   Training data under ${CNN_TRAINING_DATA}/
# output:
#   Convolutional Neural Network model: ${LYM_NECRO_CNN_MODEL_PATH}/cnn_lym_model.pkl

source ../../conf/variables.sh

THEANO_FLAGS="device=${LYM_CNN_TRAINING_DEVICE}" python -u train_cnn_lymphocyte.py \
    ${CAE_MODEL_PATH} ${LYM_CNN_TRAINING_DATA} ${LYM_NECRO_CNN_MODEL_PATH} &> ${LOG_OUTPUT_FOLDER}/log.train_cnn_lymphocyte.txt

exit 0
