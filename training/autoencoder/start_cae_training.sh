#!/bin/bash
# start training the CNN
# inputs:
#   Mean and variance of the data: models/mu.pkl, models/sigma.pkl
#   Training data under ../data/
# output:
#   Convolutional Autoencoder model: models/cae_model.pkl
#       There is an existing model. You can skip this step by:
#       cp models/cae_model_trained.pkl models/cae_model.pkl

source ../conf/variables.sh
GPU=0
THEANO_FLAGS="device=cuda${GPU}" python -u train_cae.py &> ../log/log.train_cae.txt

exit 0
