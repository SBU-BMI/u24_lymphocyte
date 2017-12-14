import pickle
import sys
import os
import urllib
import gzip
import cPickle
import h5py
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
from lasagne.objectives import mse
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import softmax
from scipy import misc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score

from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator

def load_data(mu, sigma):
    h5file = 'data/nuclei_image/test/TCGA-02-0440-01Z-00-DX1.4fef88c9-eff7-4e00-be19-d0db2871329a_appMag_20_8_6-seg.h5';
    f = h5py.File(h5file, 'r');
    X = f['/data'][...][:, 0 : 3, 9 : 41, 9 : 41];
    X = (X - mu) / sigma;
    return X.astype(np.float32);

def get_output(ae, X):
    last_layer = ae.get_all_layers()[-1];
    indices = np.arange(96, X.shape[0], 96);
    sys.stdout.flush();
    # not splitting into batches can cause a memory error
    X_batches = np.split(X, indices);
    out = [];
    for count, X_batch in enumerate(X_batches):
        out.append(last_layer.get_output(X_batch).eval());
        sys.stdout.flush();
    return np.vstack(out);

sys.setrecursionlimit(10000);
ae = pickle.load(open('model/conv_ae_17_0.pkl','rb'));
mu = pickle.load(open('model/conv_mu.pkl','rb'));
sigma = pickle.load(open('model/conv_sigma.pkl','rb'));

X = load_data(mu, sigma);

X_recon = get_output(ae, X) * sigma + mu;
X_recon = np.rint(X_recon * 255).astype(int);
X_recon = np.clip(X_recon, a_min = 0, a_max = 255);
X_recon  = X_recon.astype('uint8');
np.savetxt('data/recon.txt', X_recon, fmt = '%d');

