#!/bin/bash
# start training the CNN for necrosis segmentation
# inputs:
#   Mean and variance of the data: models/mu.pkl, models/sigma.pkl
#   Training data under ../data/
# output:
#   Convolutional Neural Network model: models/cnn_model.pkl

GPU=0
THEANO_FLAGS="device=cuda${GPU}" python -u train_cnn_necrosis.py &> ../log/log.train_cnn_necrosis.txt

exit 0
