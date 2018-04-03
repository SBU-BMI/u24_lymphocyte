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
import matplotlib.pyplot as plt
from lasagne import layers
from lasagne import regularization
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import linear
from scipy import misc
from scipy.stats import pearsonr

from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator
from smthact import SmthAct2Layer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

def load_data(classn):
    mu = pickle.load(open('model/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model/conv_sigma.pkl', 'rb'));

    X_test = np.empty(shape = (0, 3, 32, 32));
    X_val = np.empty(shape = (0, 3, 32, 32));
    X_train = np.empty(shape = (0, 3, 32, 32));

    y_test = np.empty(shape = (0, classn));
    y_val = np.empty(shape = (0, classn));
    y_train = np.empty(shape = (0, classn));

    lines = [line.rstrip('\n') for line in open('./data/image/roundness.txt')];
    for line in lines:
        img = line.split('\t')[0];
        lab = [float(x) for x in line.split('\t')[1].split()];
        png = misc.imread('./data/' + img).transpose()[0 : 3, 9 : 41, 9 : 41];
        png = np.expand_dims(png, axis=0).astype(np.float32) / 255;
        splitr = np.random.random();
        if splitr < 0.2:
            X_test = np.concatenate((X_test, png));
            y_test = np.concatenate((y_test, np.expand_dims(np.array(lab), axis = 0)));
        elif splitr >= 0.2 and splitr < 0.25:
            X_val = np.concatenate((X_val, png));
            y_val = np.concatenate((y_val, np.expand_dims(np.array(lab), axis = 0)));
        elif splitr >= 0.25:
            X_train = np.concatenate((X_train, png));
            y_train = np.concatenate((y_train, np.expand_dims(np.array(lab), axis = 0)));

    X_train = (X_train.astype(np.float32) - mu) / sigma;
    X_val = (X_val.astype(np.float32) - mu) / sigma;
    X_test = (X_test.astype(np.float32) - mu) / sigma;
    y_train = y_train.astype(np.float32);
    y_val = y_val.astype(np.float32);
    y_test = y_test.astype(np.float32);

    print "Data Loaded", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape;
    return X_train, y_train, X_val, y_val, X_test, y_test;

X_train, y_train, X_val, y_val, X_test, y_test = load_data(1);
print y_train, y_val, y_test;

