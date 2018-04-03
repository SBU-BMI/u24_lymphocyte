#!/bin/bash
# start training the CNN
# inputs:
#   Training data under ${CAE_TRAINING_DATA}/
# output:
#   Convolutional Autoencoder model: ${CAE_MODEL_PATH}/cae_model.pkl

source ../../conf/variables.sh

THEANO_FLAGS="device=${CAE_TRAINING_DEVICE}" python -u train_cae.py \
    ${CAE_MODEL_PATH} ${CAE_TRAINING_DATA} &> ${LOG_OUTPUT_FOLDER}/log.train_cae.txt

exit 0
