#!/bin/bash
# start training the CNN
# inputs:
#   Convolutional Autoencoder model: models/cae_model.pkl
#   Mean and variance of the data: models/mu.pkl, models/sigma.pkl
#   Trainind data: /data03/shared/lehhou/lym_project/data/vals/*
# output:
#   Convolutional Neural Network model: models/cnn_model.pkl

source ./bashrc_theano.sh

GPU=$1
THEANO_FLAGS="device=gpu${GPU}" python train_cnn.py > log.train_cnn.txt

exit 0
