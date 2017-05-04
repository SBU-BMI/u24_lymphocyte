#!/bin/bash
# start training the CNN
# inputs:
#   Mean and variance of the data: models/mu.pkl, models/sigma.pkl
#   Trainind data: /data03/shared/lehhou/lym_project/data/vals/*
# output:
#   Convolutional Autoencoder model: models/cae_model.pkl

source ./bashrc_theano.sh

GPU=$1
THEANO_FLAGS="device=gpu${GPU}" python train_cae.py > log.train_cae.txt

exit 0
