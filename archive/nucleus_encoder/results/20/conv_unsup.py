from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import identity
import pickle
import sys
import os
import urllib
import gzip
import cPickle
import h5py

from math import floor
from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator

def load_data():
    nread = 0;
    nfrac = 2;
    X_buf = np.zeros((500000, 3, 32, 32));
    nbuf = 0;
    X_train = np.empty(shape=(0, 3, 32, 32));
    for fn in os.listdir('data/nuclei_image/train/'):
        if fn.endswith('.h5'):
            nread += 1;
            if nread % 1000 == 0:
                print "Reading", nread;
                sys.stdout.flush();
            f = h5py.File('data/nuclei_image/train/' + fn, 'r');
            X = f['/data'][...][0::nfrac, 0:3, 9:41, 9:41];
            f.close()

            ndata = X.shape[0];
            if nbuf + ndata > 500000:
                X_train = np.concatenate((X_train, X_buf[0:nbuf, :, :, :]));
                nbuf = 0;
            X_buf[nbuf : nbuf + ndata, :, :, :] = X;
            nbuf += ndata;
    X_train = np.concatenate((X_train, X_buf[0 : nbuf, :, :, :]));
    print "Computing mean and std";
    sys.stdout.flush();
    mu = np.mean(X_train[0::floor(X_train.shape[0] / 100000), :, :, :].flatten());
    sigma = np.std(X_train[0::floor(X_train.shape[0] / 100000), :, :, :].flatten());
    X_train = (X_train - mu) / sigma;

    print "Data Loaded", X_train.shape[0];
    sys.stdout.flush();
    return X_train.astype(np.float32), mu, sigma;

X_train, mu, sigma = load_data();
X_out = X_train.reshape((X_train.shape[0], -1));

ae = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('encode_layer', layers.DenseLayer),
        ('hidden', layers.DenseLayer),  # output_dense
        ('reshape', ReshapeLayer),
        ('unpool1', Unpool2DLayer),
        ('deconv1', layers.Conv2DLayer),
        ('unpool2', Unpool2DLayer),
        ('deconv2', layers.Conv2DLayer),
        ('unpool3', Unpool2DLayer),
        ('deconv3', layers.Conv2DLayer),
        ('unpool4', Unpool2DLayer),
        ('deconv4', layers.Conv2DLayer),
        ('output_layer', ReshapeLayer),
        ],
    input_shape=(None, 3, 32, 32),
    conv1_num_filters=80,
    conv1_filter_size=(3, 3),
    conv1_border_mode="same",
    conv1_nonlinearity=identity,
    pool1_pool_size=(2, 2),
    conv2_num_filters=100,
    conv2_filter_size=(3, 3),
    conv2_border_mode="same",
    conv2_nonlinearity=identity,
    pool2_pool_size=(2, 2),
    conv3_num_filters=120,
    conv3_filter_size=(3, 3),
    conv3_border_mode="same",
    conv3_nonlinearity=identity,
    pool3_pool_size=(2, 2),
    conv4_num_filters=140,
    conv4_filter_size=(3, 3),
    conv4_border_mode="same",
    conv4_nonlinearity=identity,
    pool4_pool_size=(2, 2),

    encode_layer_num_units=100,

    hidden_num_units=2 * 2 * 140,
    reshape_shape=(([0], 140, 2, 2)),
    unpool1_ds=(2, 2),
    deconv1_num_filters=120,
    deconv1_filter_size=(3, 3),
    deconv1_border_mode="same",
    deconv1_nonlinearity=identity,
    unpool2_ds=(2, 2),
    deconv2_num_filters=100,
    deconv2_filter_size=(3, 3),
    deconv2_border_mode="same",
    deconv2_nonlinearity=identity,
    unpool3_ds=(2, 2),
    deconv3_num_filters=80,
    deconv3_filter_size=(3, 3),
    deconv3_border_mode="same",
    deconv3_nonlinearity=identity,
    unpool4_ds=(2, 2),
    deconv4_num_filters=3,
    deconv4_filter_size=(3, 3),
    deconv4_border_mode="same",
    deconv4_nonlinearity=identity,

    output_layer_shape=(([0], -1)),
    update_learning_rate=0.0005,
    update_momentum=0.975,
    batch_iterator_train=FlipBatchIterator(batch_size=96),
    regression=True,
    max_epochs=1,
    verbose=1,
);

sys.setrecursionlimit(10000);
it_div = 1;
for ep in range(100):
    for it in range(it_div):
        ae.fit(X_train[it::it_div, :, :, :], X_out[it::it_div, :]);
        sys.stdout.flush();
        pickle.dump(ae, open("model/conv_ae_{}_{}.pkl".format(ep, it), 'w'));
        pickle.dump(ae, open('model/conv_ae.pkl', 'w'));
        pickle.dump(mu, open('model/conv_mu.pkl', 'w'));
        pickle.dump(sigma, open('model/conv_sigma.pkl', 'w'));

print "Training finished";
sys.stdout.flush();

