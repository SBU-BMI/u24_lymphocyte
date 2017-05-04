from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import mse
from lasagne.nonlinearities import tanh
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

def get_output_from_nn(last_layer, X):
    indices = np.arange(96, X.shape[0], 96);
    sys.stdout.flush();

    # not splitting into batches can cause a memory error
    X_batches = np.split(X, indices);
    out = [];
    for count, X_batch in enumerate(X_batches):
        out.append(last_layer.get_output(X_batch).eval());
        sys.stdout.flush();
    return np.vstack(out);

def encode_input(encode_layer, X):
    return get_output_from_nn(encode_layer, X);

def decode_input(decode_layer, X):
    return get_output_from_nn(decode_layer, X);

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
pickle.dump(mu, open('model/conv_mu.pkl','w'));
pickle.dump(sigma, open('model/conv_sigma.pkl','w'));

