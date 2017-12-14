#!/bin/bash
# start training the CNN
# inputs:
#   Mean and variance of the data: models/mu.pkl, models/sigma.pkl
#   Trainind data: /data03/shared/lehhou/lym_project/data/vals/*
# output:
#   Convolutional Autoencoder model: models/cae_model.pkl
#       There is an existing model. You can skip this step by:
#       cp models/cae_model_trained.pkl models/cae_model.pkl

source ../conf/bashrc_theano.sh

GPU=$1
THEANO_FLAGS="device=gpu${GPU}" python -u train_cae.py > log.train_cae.txt

exit 0
