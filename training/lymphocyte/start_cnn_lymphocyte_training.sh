#!/bin/bash
# start training the CNN
# inputs:
#   Convolutional Autoencoder model: models/cae_model.pkl
#   Mean and variance of the data: models/mu.pkl, models/sigma.pkl
#   Training data under ../data/
# output:
#   Convolutional Neural Network model: models/cnn_model.pkl

GPU=0
THEANO_FLAGS="device=cuda${GPU}" python -u train_cnn_lymphocyte.py &> ../log/log.train_cnn_lymphocyte.txt

exit 0
