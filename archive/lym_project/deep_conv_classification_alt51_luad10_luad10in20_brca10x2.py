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
from extensive_data_aug_300x300_rand_deter import data_aug
from ch_inner_prod import ChInnerProd, ChInnerProdMerge

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

APS = 300;
PS = 100;
cv_v = int(sys.argv[1]);
LearningRate = theano.shared(np.array(1e-4, dtype=np.float32));
BatchSize = 100;

filename_code = 21;
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_mu.pkl'.format(filename_code);
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_sigma.pkl'.format(filename_code);

mu = pickle.load(open(filename_mu, 'rb'));
sigma = pickle.load(open(filename_sigma, 'rb'));
model_dump = 'model_vals/' + sys.argv[0];
aug_fea_n = 1;

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


def load_data_folder(classn, folder, is_train):
    X = np.zeros(shape=(30000, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(30000, classn), dtype=np.int32);

    nline = 0;
    lines = [line.rstrip('\n') for line in open(folder + '/label.txt')];
    for line in lines:
        img = line.split()[0];
        lab = np.array([int(int(line.split()[1]) > 0)]);
        png = np.array(Image.open(folder + '/' + img).convert('RGB')).transpose();
        if (png.shape[1] == 500):
            png = png[:, 100:-100, 100:-100];
        X[nline], y[nline] = png, lab;
        nline += 1;

    X = X[0:nline];
    y = y[0:nline];
    return X, y;


def load_data_split(classn, folders, is_train):
    X = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(0, classn), dtype=np.int32);
    for folder in folders:
        X_split, y_split = load_data_folder(classn, folder, is_train);
        X = np.concatenate((X, X_split));
        y = np.concatenate((y, y_split));
    return X, y;


def load_data(classn, valid_num):
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_train = np.zeros(shape=(0, classn), dtype=np.int32);

    all_data_folder = './data/vals/';
    lines = [line.rstrip('\n') for line in open(all_data_folder + '/and_agree_luad10_luad10in20_brca10x2.txt')];
    valid_i = 0;
    for line in lines:
        split_folders = [all_data_folder + s for s in line.split()];
        if valid_i == valid_num:
            X_test, y_test = load_data_split(classn, split_folders, False);
        else:
            X_split, y_split = load_data_split(classn, split_folders, True);
            X_train = np.concatenate((X_train, X_split));
            y_train = np.concatenate((y_train, y_split));
        valid_i += 1;

    print "Data Loaded", X_train.shape, y_train.shape, X_test.shape, y_test.shape;
    return X_train, y_train, X_test, y_test;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def multi_win_during_val(val_fn, inputs, targets):
    for idraw in [50, 75, 100, 125, 150]:
        for jdraw in [50, 75, 100, 125, 150]:
            inpt_multiwin = data_aug(inputs, mu, sigma, deterministic=True, idraw=idraw, jdraw=jdraw);
            err_pat, output_pat = val_fn(inpt_multiwin, targets);
            if 'weight' in locals():
                dis = ((idraw/100.0-1.0)**2 + (jdraw/100.0-1.0)**2)**0.5;
                wei = np.exp(-np.square(dis)/2.0/0.5**2);
                weight += wei;
                err += err_pat * wei;
                output += output_pat * wei;
            else:
                dis = ((idraw/100.0-1.0)**2 + (jdraw/100.0-1.0)**2)**0.5;
                weight = np.exp(-np.square(dis)/2.0/1.0**2);
                err = err_pat * weight;
                output = output_pat * weight;
    return err/weight, output/weight;


def val_fn_epoch(classn, val_fn, X_val, y_val):
    val_err = 0;
    Pr = np.empty(shape = (100000, classn), dtype = np.int32);
    Or = np.empty(shape = (100000, classn), dtype = np.float32);
    Tr = np.empty(shape = (100000, classn), dtype = np.int32);
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


def train_round(num_epochs, network, valid_num, train_fn, val_fn, classn, X_train, y_train, X_test, y_test):
    print("Starting training...");
    print("TrLoss\tVaLoss\tAUC\tCMatrix0\tCMatrix1\tCMatrix2\tEpochs\tTime");
    start_time = time.time();
    for epoch in range(num_epochs+1):
        train_err = 0;
        train_batches = 0;
        for batch in iterate_minibatches(X_train, y_train, BatchSize, shuffle = True):
            inputs, targets = batch;
            inputs = data_aug(inputs, mu, sigma);
            train_err += train_fn(inputs, targets);
            train_batches += 1;
        train_err = train_err / train_batches;

        if epoch % 1 == 0:
            # And a full pass over the validation data:
            test_err, _, _, _, Or, Tr = val_fn_epoch(classn, val_fn, X_test, y_test);
            tpos0, tneg0, fpos0, fneg0 = confusion_matrix(Or, Tr, 0.4);
            tpos1, tneg1, fpos1, fneg1 = confusion_matrix(Or, Tr, 0.5);
            tpos2, tneg2, fpos2, fneg2 = confusion_matrix(Or, Tr, 0.6);
            val_auc = auc_roc(Or, Tr);
            # Then we print the results for this epoch:
            print("{:.4f}\t{:.4f}\t{:.4f}\t{}/{}/{}/{}\t{}/{}/{}/{}\t{}/{}/{}/{}\t{}/{}\t{:.3f}".format(
                train_err, test_err, val_auc,
                tpos0, tneg0, fpos0, fneg0,
                tpos1, tneg1, fpos1, fneg1,
                tpos2, tneg2, fpos2, fneg2,
                epoch+1, num_epochs, time.time()-start_time));
            start_time = time.time();

        if epoch % 10 == 0:
            param_values = layers.get_all_param_values(network);
            pickle.dump(param_values, open(model_dump + "_e{}_cv{}.pkl".format(epoch, valid_num), 'w'));

        if epoch == 5:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 20:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 50:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));


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
    output = lasagne.layers.get_output(network, deterministic=False);
    loss = lasagne.objectives.binary_crossentropy(output, target_var).mean();

    deter_output = lasagne.layers.get_output(network, deterministic=True);
    deter_loss = lasagne.objectives.binary_crossentropy(deter_output, target_var).mean();

    params = layers.get_all_params(network, trainable=True);
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=LearningRate, momentum=0.985);

    val_fn = theano.function([input_var, target_var], [deter_loss, deter_output]);
    train_fn = theano.function([input_var, target_var], loss, updates=updates);

    return train_fn, val_fn;


def split_validation(classn, valid_num):
    X_train, y_train, X_test, y_test = load_data(classn, valid_num);

    network, input_var, target_var = build_network_from_ae(classn);
    train_fn, val_fn = make_training_functions(network, input_var, target_var);
    train_round(1024, network, valid_num, train_fn, val_fn, classn, X_train, y_train, X_test, y_test);

    return;


def main():
    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    print "DOING CV", cv_v;
    split_validation(classn, cv_v);


if __name__ == "__main__":
    main();

