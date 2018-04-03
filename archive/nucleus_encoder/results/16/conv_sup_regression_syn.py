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
from ellipse import rand_nuclei

from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator
from smthact import SmthAct2Layer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

def iterate_minibatches(inputs, targets, batchsize, shuffle = False):
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
            excerpt = indices[start_idx : start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx + batchsize : len(inputs)];
        else:
            excerpt = slice(start_idx + batchsize, len(inputs));
        yield inputs[excerpt], targets[excerpt];


def val_fn_epoch(classn, val_fn, X_val, y_val):
    val_err = 0;
    Er = np.empty(shape = (0, 100), dtype = np.float32);
    Or = np.empty(shape = (0, classn), dtype = np.float32);
    Tr = np.empty(shape = (0, classn), dtype = np.float32);
    val_batches = 0;
    for batch in iterate_minibatches(X_val, y_val, batchsize = 100, shuffle = False):
        inputs, targets = batch;
        err, encode, hidden, smth_act, output = val_fn(inputs, targets);
        val_err += err;
        Er = np.concatenate((Er, encode));
        Or = np.concatenate((Or, output));
        Tr = np.concatenate((Tr, targets));
        val_batches += 1;
    val_err = val_err / val_batches;
    return val_err, Er, Or, Tr;


def comp_rmse(Or, Tr):
    return np.mean(np.square(Or - Tr))**0.5;


def comp_pear(Or, Tr):
    return pearsonr(Or, Tr)[0][0];

def data_gen(num, classn):
    mu = pickle.load(open('model/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model/conv_sigma.pkl', 'rb'));

    X_train = np.empty(shape = (num, 3, 32, 32));
    X_test = np.empty(shape = (1000, 3, 32, 32));
    y_train = np.empty(shape = (num, classn));
    y_test = np.empty(shape = (1000, classn));

    for i in range(num):
        im, ratio = rand_nuclei(32);
        X_train[i, :, :, :] = im.transpose();
        y_train[i] = ratio;

    for i in range(1000):
        im, ratio = rand_nuclei(32);
        X_test[i, :, :, :] = im.transpose();
        y_test[i] = ratio;

    X_train = (X_train.astype(np.float32) / 255 - mu) / sigma;
    y_train = y_train.astype(np.float32);
    X_test = (X_test.astype(np.float32) / 255 - mu) / sigma;
    y_test = y_test.astype(np.float32);
    return X_train, y_train, X_test, y_test;


def train_round(train_fn, val_fn, classn):
    print("Starting training...");
    print("TrLoss\t\tVaLoss\t\tVaRMSE\t\tVaPear\t\tEpochs\t\tTime");
    num_epochs = 1000;
    batchsize = 100;
    start_time = time.time();
    for epoch in range(num_epochs):
        if epoch % 50 == 0:
            X_train, y_train, X_val, y_val = data_gen(4000, classn);
        train_err = 0;
        train_batches = 0;
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle = True):
            inputs, targets = batch;
            train_err += train_fn(inputs, targets);
            train_batches += 1;
        train_err = train_err / train_batches;

        if epoch % 10 == 0:
            val_err, Er, Or, Tr = val_fn_epoch(classn, val_fn, X_val, y_val);
            print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}\t\t{:.3f}".format(
                train_err, val_err, comp_rmse(Or, Tr), comp_pear(Or, Tr), epoch + 1, num_epochs, time.time() - start_time));
            start_time = time.time();


def stack_pretrain_round(stack_train_fn, val_fn, classn):
    print("Starting stack_pretraining...");
    print("TrLoss\t\tVaLoss\t\tVaRMSE\t\tVaPear\t\tEpochs\t\tTime");
    stack_num_epochs = 300;
    stack_batchsize = 100;
    start_time = time.time();
    for epoch in range(stack_num_epochs):
        if epoch % 50 == 0:
            X_train, y_train, X_val, y_val = data_gen(4000, classn);
        train_err = 0;
        train_batches = 0;
        for batch in iterate_minibatches(X_train, y_train, stack_batchsize, shuffle = True):
            inputs, targets = batch;
            train_err += stack_train_fn(inputs, targets);
            train_batches += 1;
        train_err = train_err / train_batches;

        if epoch % 10 == 0:
            val_err, Er, Or, Tr = val_fn_epoch(classn, val_fn, X_val, y_val);
            print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}\t\t{:.3f}".format(
                train_err, val_err, comp_rmse(Or, Tr), comp_pear(Or, Tr), epoch + 1, stack_num_epochs, time.time() - start_time));
            start_time = time.time();


def build_network_from_ae(classn):
    input_var = T.tensor4('inputs');
    target_var = T.matrix('targets');

    ae = pickle.load(open('model/conv_ae.pkl', 'rb'));

    input_layer_index = map(lambda pair : pair[0], ae.layers).index('input');
    first_layer = ae.get_all_layers()[input_layer_index + 1];
    input_layer = layers.InputLayer(shape = (None, 3, 32, 32), input_var = input_var);
    first_layer.input_layer = input_layer;

    encode_layer_index = map(lambda pair : pair[0], ae.layers).index('encode_layer');
    encode_layer = ae.get_all_layers()[encode_layer_index];

    # conventional recitified linear units
    #hidden_layer = layers.DenseLayer(incoming = encode_layer, num_units = 200, nonlinearity = rectify);
    #network = layers.DenseLayer(incoming = hidden_layer, num_units = classn, nonlinearity = linear);
    #stack_params = [network.W, network.b, hidden_layer.W, hidden_layer.b];

    # smooth activation function
    hidden_layer = layers.DenseLayer(incoming = encode_layer, num_units = 200, nonlinearity = linear);
    smth_act_layer = SmthAct2Layer(incoming = hidden_layer, x_start = -10.0, x_end = 10.0, num_segs = 20);
    network = layers.DenseLayer(incoming = smth_act_layer, num_units = classn, nonlinearity = linear);
    stack_params = [network.W, network.b, hidden_layer.W, hidden_layer.b, smth_act_layer.W];

    return (encode_layer, hidden_layer, smth_act_layer, network), input_var, target_var, stack_params;


def make_training_functions(network_layers, input_var, target_var, stack_params, weight_decay):
    encode_layer, hidden_layer, smth_act_layer, network = network_layers;

    output = lasagne.layers.get_output(network, deterministic = True);
    loss = lasagne.objectives.squared_error(output, target_var).mean() + \
           weight_decay * regularization.regularize_network_params(
                   layer = network, penalty = regularization.l2, tags={'regularizable' : True});

    params = layers.get_all_params(network, trainable = True);
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = 0.00002, momentum = 0.95);
    stack_updates = lasagne.updates.nesterov_momentum(loss, stack_params, learning_rate = 0.00001, momentum = 0.95);

    encode = lasagne.layers.get_output(encode_layer, deterministic = True);
    hidden = lasagne.layers.get_output(hidden_layer, deterministic = True);
    smth_act = lasagne.layers.get_output(smth_act_layer, deterministic = True);

    val_fn = theano.function([input_var, target_var], [loss, encode, hidden, smth_act, output]);
    train_fn = theano.function([input_var, target_var], loss, updates = updates);
    stack_train_fn = theano.function([input_var, target_var], loss, updates = stack_updates);

    return val_fn, train_fn, stack_train_fn;


def save_result(cache_str, Er, Or, Tr):
    cache_dir = './cache/' + cache_str + '/';
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir);
    np.save(cache_dir + 'Er', Er);
    np.save(cache_dir + 'Or', Or);
    np.save(cache_dir + 'Tr', Tr);


def load_result(cache_str):
    cache_dir = './cache/' + cache_str + '/';
    Er = np.load(cache_dir + 'Er.npy');
    Or = np.load(cache_dir + 'Or.npy');
    Tr = np.load(cache_dir + 'Tr.npy');
    return Er, Or, Tr;


def save_visual_cases(X, Or, Tr):
    mu = pickle.load(open('model/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model/conv_sigma.pkl', 'rb'));

    if not os.path.isfile('./visual_regression/truth.txt'):
        with open('./visual_regression/truth.txt', 'w'):
            pass;

    ind = sum(1 for line in open('./visual_regression/truth.txt'));
    for xi in range(X.shape[0]):
        ind += 1;
        im = ((X[xi].transpose() * sigma + mu) * 255).astype(np.uint8);
        misc.imsave("./visual_regression/case_{}.png".format(ind), im);

    f = file('./visual_regression/output.txt', 'a');
    np.savetxt(f, Or, fmt='%.4f');
    f.close();

    f = file('./visual_regression/truth.txt', 'a');
    np.savetxt(f, Tr, fmt='%.4f');
    f.close();


def split_validation(classn, weight_decay):
    network_layers, input_var, target_var, stack_params = build_network_from_ae(classn);
    val_fn, train_fn, stack_train_fn = make_training_functions(network_layers, input_var, target_var, stack_params, weight_decay);

    stack_pretrain_round(stack_train_fn, val_fn, classn);
    train_round(train_fn, val_fn, classn);

    # Testing
    X_test, y_test, _, _ = data_gen(4000, classn);
    _, Er, Or, Tr = val_fn_epoch(classn, val_fn, X_test, y_test);
    save_visual_cases(X_test, Or, Tr);
    return Er, Or, Tr;


def main():
    classn = 1;
    split_n = 1;
    weight_decay = 0.00002;
    sys.setrecursionlimit(10000);

    for v in range(split_n):
        Er, Or, Tr = split_validation(classn, weight_decay);
        save_result("regression_sp_{}".format(v), Er, Or, Tr);
        print("[CNN] RMSE: {:.4f} Pear: {:.4f}".format(comp_rmse(Or, Tr), comp_pear(Or, Tr)));


if __name__ == "__main__":
    main();

