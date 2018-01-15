import pickle
import sys
import os
import urllib
import gzip
import cPickle
import time
import lasagne
import theano
import numpy as np
import theano.tensor as T

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import sigmoid, rectify, leaky_rectify, identity
from lasagne.nonlinearities import softmax
from lasagne import regularization
from scipy import misc
from PIL import Image
from lasagne import init
from math import floor

from shape import ReshapeLayer
from batch_norms import batch_norm, SoftThresPerc
from extensive_data_aug_100x100 import data_aug
from ch_inner_prod import ChInnerProd, ChInnerProdMerge

PS = 100;
LearningRate = theano.shared(np.array(3e-2, dtype=np.float32));
NumEpochs = 100;
BatchSize = 32;
filename_code = 28;
filename_model_ae = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_model_{}.pkl'.format(filename_code, '{}');
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_mu.pkl'.format(filename_code);
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_sigma.pkl'.format(filename_code);

def load_data():
    nbuf = 0;
    X_train = np.zeros(shape=(500000, 3, 100, 100), dtype=np.float32);
    lines = [line.rstrip('\n') for line in open('./data/vals/random_patches_for_all_svs/label.txt')];
    for line in lines:
        full_path = './data/vals/random_patches_for_all_svs/image_' + line.split()[0];
        png = np.array(Image.open(full_path).convert('RGB')).transpose() / 255.0;
        X_train[nbuf, :, :, :] = png;
        nbuf += 1;

    X_train = X_train[0:nbuf];
    print "Computing mean and std";
    mu = np.mean(X_train[0::int(floor(X_train.shape[0]/1000)), :, :, :].flatten());
    sigma = np.std(X_train[0::int(floor(X_train.shape[0]/1000)), :, :, :].flatten());
    X_train = (X_train - mu) / sigma;

    print "Data Loaded", X_train.shape[0];
    return X_train, mu, sigma;


def iterate_minibatches_ae(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs));
        np.random.shuffle(indices);

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt];


def build_autoencoder_network():
    input_var = T.tensor4('input_var');

    layer = layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);
    layer = batch_norm(layers.Conv2DLayer(layer, 100,  filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 120,  filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = layers.Pool2DLayer(layer, pool_size=(2,2), stride=2, mode='average_inc_pad');
    layer = batch_norm(layers.Conv2DLayer(layer, 240,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = layers.Pool2DLayer(layer, pool_size=(2,2), stride=2, mode='average_inc_pad');
    layer = batch_norm(layers.Conv2DLayer(layer, 640,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    prely = batch_norm(layers.Conv2DLayer(layer, 1024, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));

    featm    = batch_norm(layers.Conv2DLayer(prely, 640, filter_size=(1,1), nonlinearity=leaky_rectify));
    feat_map = batch_norm(layers.Conv2DLayer(featm, 100, filter_size=(1,1), nonlinearity=rectify, name="feat_map"));
    mask_map = feat_map;
    layer    = feat_map;

    layer = batch_norm(layers.Deconv2DLayer(layer, 1024, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 640,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 640,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 320,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 320,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 240,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 120,  filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 100,  filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer =            layers.Deconv2DLayer(layer, 3,    filter_size=(1,1), stride=1, crop='same', nonlinearity=identity);

    glblf = batch_norm(layers.Conv2DLayer(prely, 128,  filter_size=(1,1), nonlinearity=leaky_rectify));
    glblf = layers.Pool2DLayer(glblf, pool_size=(5,5), stride=5, mode='average_inc_pad');
    glblf = batch_norm(layers.Conv2DLayer(glblf, 64,   filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Conv2DLayer(glblf, 5,    filter_size=(1,1), nonlinearity=rectify), name="global_feature");

    glblf = batch_norm(layers.Deconv2DLayer(glblf, 256, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(9,9), stride=5, crop=(2,2),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf =            layers.Deconv2DLayer(glblf, 3,   filter_size=(1,1), stride=1, crop='same', nonlinearity=identity);

    layer = layers.ElemwiseSumLayer([layer, glblf]);

    network = ReshapeLayer(layer, ([0], -1));
    mask_var = lasagne.layers.get_output(mask_map);
    output_var = lasagne.layers.get_output(network);

    return network, input_var, mask_var, output_var;


def build_training_function(network, input_var, mask_var, output_var):
    print("building training function");

    target_var = T.matrix('target_var');
    loss = lasagne.objectives.squared_error(output_var, target_var).mean();

    param_set = lasagne.layers.get_all_params(network, trainable=True);
    updates = lasagne.updates.nesterov_momentum(loss, param_set, learning_rate=LearningRate, momentum=0.9);
    train_func = theano.function([input_var, target_var], [loss, mask_var], updates=updates);

    print("finish building training function");
    return train_func;


def exc_train(train_func, X_train, network):
    print("Starting training...");
    print("Epoch\t\tIter\t\tLoss\t\tSpar\t\tTime");
    it_div = 100;
    for epoch in range(NumEpochs):
        start_time = time.time();
        for it in range(it_div):
            # Iterate through mini batches
            total_loss = 0;
            total_sparsity = 0;
            n_batch = 0;
            for batch in iterate_minibatches_ae(X_train[it::it_div], BatchSize, shuffle=True):
                batch = data_aug(batch);
                batch_target = np.reshape(batch, (batch.shape[0], -1));
                loss, mask = train_func(batch, batch_target);
                total_loss += loss;
                total_sparsity += 100.0 * float(np.count_nonzero(mask>1e-6)) / mask.size;
                n_batch += 1;
            total_loss /= n_batch;
            total_sparsity /= n_batch;
            LearningRate.set_value(np.float32(0.99*LearningRate.get_value()));

            print("{:d}\t\t{:d}\t\t{:.4f}\t\t{:.3f}\t\t{:.3f}".format(
                epoch, it, total_loss, total_sparsity, time.time()-start_time));
            start_time = time.time();

        if epoch % 1 == 0:
            param_values = layers.get_all_param_values(network);
            pickle.dump(param_values, open(filename_model_ae.format(epoch), 'w'));


def main():
    X_train, mu, sigma = load_data();

    pickle.dump(mu, open(filename_mu, 'w'));
    pickle.dump(sigma, open(filename_sigma, 'w'));

    # Build network
    network, input_var, mask_var, output_var = build_autoencoder_network();
    train_func = build_training_function(network, input_var, mask_var, output_var);
    exc_train(train_func, X_train, network);

    print("DONE !");


if __name__ == "__main__":
    main();

