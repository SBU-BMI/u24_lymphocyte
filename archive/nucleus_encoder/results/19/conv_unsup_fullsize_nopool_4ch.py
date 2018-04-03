from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import rectify
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
    nfrac = 10;
    nbuf = 0;
    X_buf = np.zeros((500000, 4, 50, 50), dtype = np.float32);
    X_train = np.empty(shape=(0, 4, 50, 50), dtype = np.float32);
    for fn in os.listdir('data/nuclei_image/train/'):
        if fn.endswith('.h5'):
            nread += 1;
            if nread % 1000 == 0:
                print "Reading", nread;
            f = h5py.File('data/nuclei_image/train/' + fn, 'r');
            X = f['/data'][...][0::nfrac, 0:4, :, :];
            f.close()

            ndata = X.shape[0];
            if nbuf + ndata > 500000:
                X_train = np.concatenate((X_train, X_buf[0:nbuf, :, :, :]));
                nbuf = 0;
            X_buf[nbuf : nbuf + ndata, :, :, :] = X.astype(np.float32);
            nbuf += ndata;
    X_train = np.concatenate((X_train, X_buf[0 : nbuf, :, :, :]));
    print "Computing mean and std";
    mu = np.mean(X_train[0::floor(X_train.shape[0] / 10000), :, :, :].flatten());
    sigma = np.std(X_train[0::floor(X_train.shape[0] / 10000), :, :, :].flatten());
    X_train = (X_train - mu) / sigma;

    print "Data Loaded", X_train.shape[0];
    return X_train, mu, sigma;

if os.path.exists('model_fullsize_nopool_4ch/X_train.pkl'):
    X_train = pickle.load(open('model_fullsize_nopool_4ch/X_train.pkl', 'rb'));
    X_out = pickle.load(open('model_fullsize_nopool_4ch/X_out.pkl', 'rb'));
    print "Data Loaded", X_train.shape[0];
else:
    X_train, mu, sigma = load_data();
    X_out = X_train.reshape((X_train.shape[0], -1));
    pickle.dump(X_train, open('model_fullsize_nopool_4ch/X_train.pkl', 'w'));
    pickle.dump(X_train, open('model_fullsize_nopool_4ch/X_out.pkl', 'w'));
    pickle.dump(mu, open('model_fullsize_nopool_4ch/conv_mu.pkl', 'w'));
    pickle.dump(sigma, open('model_fullsize_nopool_4ch/conv_sigma.pkl', 'w'));
    print "Data Dumped";

ae = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('drop1', layers.DropoutLayer),

        ('conv1', layers.Conv2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('conv3', layers.Conv2DLayer),

        ('pool1', layers.MaxPool2DLayer),

        ('conv4', layers.Conv2DLayer),
        ('conv5', layers.Conv2DLayer),
        ('conv6', layers.Conv2DLayer),

        ('pool2', layers.MaxPool2DLayer),

        ('pre_encode', layers.DenseLayer),
        ('encode_layer', layers.DenseLayer),
        ('pos_encode', layers.DenseLayer),

        ('hidden', layers.DenseLayer),
        ('reshape', ReshapeLayer),

        ('unpool1', Unpool2DLayer),

        ('deconv1', layers.Conv2DLayer),
        ('deconv2', layers.Conv2DLayer),
        ('deconv3', layers.Conv2DLayer),

        ('unpool2', Unpool2DLayer),

        ('deconv4', layers.Conv2DLayer),
        ('deconv5', layers.Conv2DLayer),
        ('deconv6', layers.Conv2DLayer),

        ('output_layer', ReshapeLayer),
        ],
    input_shape=(None, 4, 50, 50),
    drop1_p=0.08,

    conv1_num_filters=80,
    conv1_filter_size=(4, 4),
    conv2_num_filters=80,
    conv2_filter_size=(4, 4),
    conv3_num_filters=120,
    conv3_filter_size=(3, 3),

    pool1_pool_size=(2, 2),

    conv4_num_filters=100,
    conv4_filter_size=(4, 4),
    conv5_num_filters=140,
    conv5_filter_size=(3, 3),
    conv6_num_filters=140,
    conv6_filter_size=(3, 3),

    pool2_pool_size=(2, 2),

    pre_encode_num_units=500,
    encode_layer_num_units=200,
    pos_encode_num_units=500,

    hidden_num_units=18 * 18 * 140,
    reshape_shape=(([0], 140, 18, 18)),

    unpool1_ds=(2, 2),

    deconv1_num_filters=140,
    deconv1_filter_size=(3, 3),
    deconv2_num_filters=100,
    deconv2_filter_size=(3, 3),
    deconv3_num_filters=120,
    deconv3_filter_size=(4, 4),

    unpool2_ds=(2, 2),

    deconv4_num_filters=80,
    deconv4_filter_size=(3, 3),
    deconv5_num_filters=80,
    deconv5_filter_size=(4, 4),
    deconv6_num_filters=4,
    deconv6_filter_size=(4, 4),

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
        pickle.dump(ae, open("model_fullsize_nopool_4ch/conv_ae_{}_{}.pkl".format(ep, it), 'w'));
        pickle.dump(ae, open('model_fullsize_nopool_4ch/conv_ae.pkl', 'w'));
        pickle.dump(mu, open('model_fullsize_nopool_4ch/conv_mu.pkl', 'w'));
        pickle.dump(sigma, open('model_fullsize_nopool_4ch/conv_sigma.pkl', 'w'));

print "Training finished";

