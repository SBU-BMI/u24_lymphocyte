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
from extensive_data_aug_100x100_heatmap import data_aug
from ch_inner_prod import ChInnerProd, ChInnerProdMerge

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

APS = 100;
PS = 100;
TileFolder = sys.argv[1] + '/';
BatchSize = 100;

filename_code = 21;
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_mu.pkl'.format(filename_code);
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_sigma.pkl'.format(filename_code);

mu = pickle.load(open(filename_mu, 'rb'));
sigma = pickle.load(open(filename_sigma, 'rb'));
aug_fea_n = 1;

read_epoch = 180;
read_cv = 0;
heat_map_out = 'patch-level-pred51.txt';
CNNModel = 'model_vals/' + sys.argv[0].split('_heatmap.py')[0] + '.py_e{}_cv{}.pkl'.format(read_epoch, read_cv);

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets);
    if inputs.shape[0] <= batchsize:
        yield inputs, targets;
        return;

    if shuffle:
        indices = np.arange(len(inputs));
        np.random.shuffle(indices);
    start_idx = 0;
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx + batchsize : len(inputs)];
        else:
            excerpt = slice(start_idx + batchsize, len(inputs));
        yield inputs[excerpt], targets[excerpt];


def load_data():
    X = np.zeros(shape=(1000000, 3, APS, APS), dtype=np.float32);
    coor = np.zeros(shape=(1000000, 2), dtype=np.int32);

    ind = 0;
    for fn in os.listdir(TileFolder):
        full_fn = TileFolder + '/' + fn;
        if not os.path.isfile(full_fn):
            continue;
        if len(fn.split('_')) < 4:
            continue;

        x_off = float(fn.split('_')[0]);
        y_off = float(fn.split('_')[1]);
        svs_pw = float(fn.split('_')[2]);
        png_pw = float(fn.split('_')[3].split('.png')[0]);

        png = np.array(Image.open(full_fn).convert('RGB'));
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue;
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue;
                X[ind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose();
                coor[ind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                coor[ind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                if ind % 1000 == 0:
                    print ind;
                ind += 1;

    X = X[0:ind];
    coor = coor[0:ind];

    print "Data Loaded", X.shape, coor.shape;
    return X, coor;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def multi_win_during_val(val_fn, inputs, targets):
    for idraw in [-1,]:
        for jdraw in [-1, -1]:
            inpt_multiwin = data_aug(inputs, mu, sigma, deterministic=True, idraw=idraw, jdraw=jdraw);
            err_pat, output_pat = val_fn(inpt_multiwin, targets);
            if 'weight' in locals():
                weight += 1.0;
                err += err_pat;
                output += output_pat;
            else:
                weight = 1.0;
                err = err_pat;
                output = output_pat;
    return err/weight, output/weight;


def val_fn_epoch(classn, val_fn, X_val, y_val):
    val_err = 0;
    Pr = np.empty(shape = (1000000, classn), dtype = np.int32);
    Or = np.empty(shape = (1000000, classn), dtype = np.float32);
    Tr = np.empty(shape = (1000000, classn), dtype = np.int32);
    val_batches = 0;
    nline = 0;
    for batch in iterate_minibatches(X_val, y_val, BatchSize, shuffle = False):
        inputs, targets = batch;
        err, output = multi_win_during_val(val_fn, inputs, targets);
        pred = from_output_to_pred(output);
        val_err += err;
        Pr[nline:nline+len(output)] = pred;
        Or[nline:nline+len(output)] = output;
        Tr[nline:nline+len(output)] = targets;
        val_batches += 1;
        nline += len(output);
    Pr = Pr[:nline];
    Or = Or[:nline];
    Tr = Tr[:nline];
    val_err = val_err / val_batches;
    val_ham = (1 - hamming_loss(Tr, Pr));
    val_acc = accuracy_score(Tr, Pr);
    return val_err, val_ham, val_acc, Pr, Or, Tr;


def confusion_matrix(Or, Tr, thres):
    tpos = np.sum((Or>=thres) * (Tr==1));
    tneg = np.sum((Or< thres) * (Tr==0));
    fpos = np.sum((Or>=thres) * (Tr==0));
    fneg = np.sum((Or< thres) * (Tr==1));
    return tpos, tneg, fpos, fneg;


def auc_roc(Pr, Tr):
    fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0);
    return auc(fpr, tpr);


def build_network_from_ae(classn):
    input_var = T.tensor4('input_var');
    target_var = T.imatrix('targets');

    layer = layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);
    layer = batch_norm(layers.Conv2DLayer(layer, 100,  filter_size=(5,5), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 100,  filter_size=(5,5), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 120,  filter_size=(4,4), stride=1, nonlinearity=leaky_rectify));
    layer = layers.MaxPool2DLayer(layer, pool_size=(3,3), stride=2);
    layer = batch_norm(layers.Conv2DLayer(layer, 240,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 480,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 480,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 480,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 480,  filter_size=(3,3), stride=1, nonlinearity=leaky_rectify));

    layer = layers.Pool2DLayer(layer, pool_size=(20,20), stride=20, mode='average_inc_pad');
    network = layers.DenseLayer(layer, classn, nonlinearity=sigmoid);

    return network, input_var, target_var;

def make_training_functions(network, input_var, target_var):
    deter_output = lasagne.layers.get_output(network, deterministic=True);
    deter_loss = lasagne.objectives.binary_crossentropy(deter_output, target_var).mean();

    val_fn = theano.function([input_var, target_var], [deter_loss, deter_output]);

    return val_fn;


def split_validation(classn):
    X, coor = load_data();

    network, input_var, target_var = build_network_from_ae(classn);
    val_fn = make_training_functions(network, input_var, target_var);
    layers.set_all_param_values(network, pickle.load(open(CNNModel, 'rb')));

    Y = np.zeros((X.shape[0], classn), dtype=np.int32);

    # Testing
    _, _, _, Pr, Or, Tr = val_fn_epoch(classn, val_fn, X, Y);

    fid = open(TileFolder + '/' + heat_map_out, 'w');
    for idx in range(0, Or.shape[0]):
        fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or[idx][0]));
    fid.close();

    return Pr, Or, Tr;


def main():
    check_ext_file = TileFolder + '/' + heat_map_out;
    if os.path.isfile(check_ext_file):
        exit(0);

    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    Pr, Or, Tr = split_validation(classn);
    print("[CNN] Hamming: {:.4f}\tAccuracy: {:.4f}".format(1 - hamming_loss(Tr, Pr), accuracy_score(Tr, Pr)));


if __name__ == "__main__":
    main();

