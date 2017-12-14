from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import mse
from lasagne.nonlinearities import rectify
import pickle
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
import os
import urllib
import gzip
import cPickle
import h5py

from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator

def load_data():
    first = True;
    for fn in os.listdir('data/nuclei_image/train/'):
        if fn.endswith('.h5'):
            f = h5py.File('data/nuclei_image/train/' + fn, 'r');
            X = f['/data'][...][:, 0 : 3, 9 : 41, 9 : 41];
            f.close()
            if first:
                X_train = X;
                first = False;
            else:
                X_train = np.concatenate((X_train, X));

    first = True;
    for fn in os.listdir('data/nuclei_image/test/'):
        if fn.endswith('.h5'):
            f = h5py.File('data/nuclei_image/test/' + fn, 'r');
            X = f['/data'][...][:, 0 : 3, 9 : 41, 9 : 41];
            f.close()
            if first:
                X_test = X;
                first = False;
            else:
                X_test = np.concatenate((X_test, X));

    mu, sigma = np.mean(X_train.flatten()), np.std(X_train.flatten());
    X_train = (X_train - mu) / sigma;
    X_test = (X_test - mu) / sigma;

    return X_train.astype(np.float32), X_test.astype(np.float32), mu, sigma;

X_train, X_test, mu, sigma = load_data();
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
    conv1_nonlinearity=rectify,
    pool1_pool_size=(2, 2),
    conv2_num_filters=100,
    conv2_filter_size=(3, 3),
    conv2_border_mode="same",
    conv2_nonlinearity=rectify,
    pool2_pool_size=(2, 2),
    conv3_num_filters=120,
    conv3_filter_size=(3, 3),
    conv3_border_mode="same",
    conv3_nonlinearity=rectify,
    pool3_pool_size=(2, 2),
    conv4_num_filters=140,
    conv4_filter_size=(3, 3),
    conv4_border_mode="same",
    conv4_nonlinearity=rectify,
    pool4_pool_size=(2, 2),

    encode_layer_num_units=100,

    hidden_num_units=2 * 2 * 140,
    reshape_shape=(([0], 140, 2, 2)),
    unpool1_ds=(2, 2),
    deconv1_num_filters=120,
    deconv1_filter_size=(3, 3),
    deconv1_border_mode="same",
    deconv1_nonlinearity=rectify,
    unpool2_ds=(2, 2),
    deconv2_num_filters=100,
    deconv2_filter_size=(3, 3),
    deconv2_border_mode="same",
    deconv2_nonlinearity=rectify,
    unpool3_ds=(2, 2),
    deconv3_num_filters=80,
    deconv3_filter_size=(3, 3),
    deconv3_border_mode="same",
    deconv3_nonlinearity=rectify,
    unpool4_ds=(2, 2),
    deconv4_num_filters=3,
    deconv4_filter_size=(3, 3),
    deconv4_border_mode="same",
    deconv4_nonlinearity=rectify,

    output_layer_shape=(([0], -1)),
    update_learning_rate=0.001,
    update_momentum=0.975,
    batch_iterator_train=FlipBatchIterator(batch_size=96),
    regression=True,
    max_epochs=200,
    verbose=1,
    );
ae.fit(X_train, X_out);
print 'Training finished';

sys.setrecursionlimit(10000);
pickle.dump(ae, open('model/conv_ae.pkl','w'));
pickle.dump(mu, open('model/conv_mu.pkl','w'));
pickle.dump(sigma, open('model/conv_sigma.pkl','w'));

