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
BatchSize = 64;
filename_code = 26;
filename_model_ae = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_model_{}.pkl'.format(filename_code, 0);
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_mu.pkl'.format(filename_code);
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_sigma.pkl'.format(filename_code);

def load_data():
    nbuf = 0;
    X_train = np.zeros(shape=(500000, 3, 100, 100), dtype=np.float32);
    lines = [line.rstrip('\n') for line in open('./data/vals/random_patches_for_all_svs/label.txt')];
    for line in lines[::1000]:
        full_path = './data/vals/random_patches_for_all_svs/image_' + line.split()[0];
        png = np.array(Image.open(full_path).convert('RGB')).transpose() / 255.0;
        X_train[nbuf, :, :, :] = png;
        nbuf += 1;

    X_train = X_train[0:nbuf];
    print "Loading mean and std";
    mu = pickle.load(open(filename_mu, 'rb'));
    sigma = pickle.load(open(filename_sigma, 'rb'));
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
    layer = batch_norm(layers.Conv2DLayer(layer,  80, filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer,  80, filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer,  80, filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer,  80, filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 100, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 100, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 100, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    prely = batch_norm(layers.Conv2DLayer(layer, 100, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));

    featm    = batch_norm(layers.Conv2DLayer(prely, 180, filter_size=(1,1), nonlinearity=leaky_rectify));
    feat_map = batch_norm(layers.Conv2DLayer(featm, 120, filter_size=(1,1), nonlinearity=rectify, name="feat_map"));
    maskm    = batch_norm(layers.Conv2DLayer(prely, 120, filter_size=(1,1), nonlinearity=leaky_rectify));
    mask_rep = batch_norm(layers.Conv2DLayer(maskm,   1, filter_size=(1,1), nonlinearity=None),   beta=None, gamma=None);
    mask_map = SoftThresPerc(mask_rep, perc=99.9, alpha=0.5, beta=init.Constant(0.5), tight=100.0, name="mask_map");
    layer    = ChInnerProdMerge(feat_map, mask_map, name="encoder");

    layer = batch_norm(layers.Deconv2DLayer(layer, 100, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 100, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 100, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 100, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer,  80, filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer,  80, filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer,  80, filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer,  80, filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer =            layers.Deconv2DLayer(layer,   3, filter_size=(1,1), stride=1, crop='same', nonlinearity=identity);

    glblf = batch_norm(layers.Conv2DLayer(prely,  100, filter_size=(1,1), nonlinearity=leaky_rectify));
    glblf = layers.Pool2DLayer(glblf, pool_size=(5,5), stride=5, mode='average_inc_pad');
    glblf = batch_norm(layers.Conv2DLayer(glblf,   64, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Conv2DLayer(glblf,    3, filter_size=(1,1), nonlinearity=rectify), name="global_feature");

    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64, filter_size=(9,9), stride=5, crop=(2,2),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 48, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 48, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 48, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf =            layers.Deconv2DLayer(glblf,  3, filter_size=(1,1), stride=1, crop='same', nonlinearity=identity);

    layer = layers.ElemwiseSumLayer([layer, glblf]);

    network = ReshapeLayer(layer, ([0], -1));
    layers.set_all_param_values(network, pickle.load(open(filename_model_ae, 'rb')));
    feat_var = lasagne.layers.get_output(feat_map, deterministic=True);
    mask_var = lasagne.layers.get_output(mask_map, deterministic=True);
    outp_var = lasagne.layers.get_output(network,  deterministic=True);

    return network, input_var, feat_var, mask_var, outp_var;


def build_testing_function(network, input_var, feat_var, mask_var, outp_var):
    print("building testing function");
    encode_decode_func = theano.function([input_var], [feat_var, mask_var, outp_var]);
    print("finish building testing function");
    return encode_decode_func;


def save_visual(all_imag, all_feat, all_mask, all_outp, mu, sigma):
    for xi in range(all_imag.shape[0]):
        imag = ((all_imag[xi].transpose() * sigma + mu) * 255.0).astype(np.uint8);
        outp = np.clip(((all_outp[xi].transpose() * sigma + mu) * 255.0), 0.1, 254.9).astype(np.uint8);
        mask = (all_mask[xi].transpose() * 255.0).astype(np.uint8);
        misc.imsave("./visual_autoencoder/imag_{}.png".format(xi), np.squeeze(imag));
        misc.imsave("./visual_autoencoder/outp_{}.png".format(xi), np.squeeze(outp));
        misc.imsave("./visual_autoencoder/mask_{}.png".format(xi), np.squeeze(mask));


def exc_test(encode_decode_func, X_train):
    all_imag = np.zeros(shape=(10000, 3,   PS, PS), dtype=np.float32);
    all_feat = np.zeros(shape=(10000, 120, PS, PS), dtype=np.float32);
    all_mask = np.zeros(shape=(10000, 1,   PS, PS), dtype=np.float32);
    all_outp = np.zeros(shape=(10000, 3,   PS, PS), dtype=np.float32);
    buf_n = 0;

    print("Starting testing...");
    # Iterate through mini batches
    for batch in iterate_minibatches_ae(X_train, BatchSize, shuffle=True):
        batch = data_aug(batch, deterministic=True);
        feat, mask, outp = encode_decode_func(batch);
        all_imag[buf_n:buf_n+BatchSize, :, :, :] = batch;
        all_feat[buf_n:buf_n+BatchSize, :, :, :] = feat;
        all_mask[buf_n:buf_n+BatchSize, :, :, :] = mask;
        all_outp[buf_n:buf_n+BatchSize, :, :, :] = outp.reshape((BatchSize, 3, PS, PS));
        buf_n += BatchSize;
    all_imag = all_imag[0:buf_n];
    all_feat = all_feat[0:buf_n];
    all_mask = all_mask[0:buf_n];
    all_outp = all_outp[0:buf_n];

    return all_imag, all_feat, all_mask, all_outp;


def main():
    X_train, mu, sigma = load_data();

    # Build network
    network, input_var, feat_var, mask_var, outp_var = build_autoencoder_network();
    encode_decode_func = build_testing_function(network, input_var, feat_var, mask_var, outp_var);
    all_imag, all_feat, all_mask, all_outp = exc_test(encode_decode_func, X_train);
    save_visual(all_imag, all_feat, all_mask, all_outp, mu, sigma);

    print("DONE !");


if __name__ == "__main__":
    main();

