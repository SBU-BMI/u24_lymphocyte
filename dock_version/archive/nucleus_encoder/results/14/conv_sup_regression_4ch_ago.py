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
from smthact import AgoLayer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

def iterate_minibatches(inputs, augs, targets, batchsize, shuffle = False):
    assert len(inputs) == len(targets);
    assert len(inputs) == len(augs);
    if inputs.shape[0] <= batchsize:
        yield inputs, augs, targets;
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
        yield inputs[excerpt], augs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx + batchsize : len(inputs)];
        else:
            excerpt = slice(start_idx + batchsize, len(inputs));
        yield inputs[excerpt], augs[excerpt], targets[excerpt];


def data_aug(X):
    bs = X.shape[0];
    h_indices = np.random.choice(bs, bs / 2, replace = False);  # horizontal flip
    v_indices = np.random.choice(bs, bs / 2, replace = False);  # vertical flip
    r_indices = np.random.choice(bs, bs / 2, replace = False);  # 90 degree rotation

    X[h_indices] = X[h_indices, :, :, ::-1];
    X[v_indices] = X[v_indices, :, ::-1, :];
    for rot in range(np.random.randint(3) + 1):
        X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3);

    return X;


def load_data(classn):
    mu = pickle.load(open('model_4ch/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model_4ch/conv_sigma.pkl', 'rb'));

    X_test = np.empty(shape = (0, 4, 32, 32));
    X_val = np.empty(shape = (0, 4, 32, 32));
    X_train = np.empty(shape = (0, 4, 32, 32));

    y_test = np.empty(shape = (0, classn));
    y_val = np.empty(shape = (0, classn));
    y_train = np.empty(shape = (0, classn));

    lines = [line.rstrip('\n') for line in open('./data/image/roundness.txt')];
    for line in lines:
        img = line.split('\t')[0].split('/')[1];
        lab = [float(x) for x in line.split('\t')[1].split()];

        mask = misc.imread('./data/mask/' + img).transpose()[9 : 41, 9 : 41];
        png = misc.imread('./data/image/' + img).transpose()[0 : 3, 9 : 41, 9 : 41];
        png_4ch = np.concatenate((png, np.expand_dims(np.array(mask), axis = 0)));
        png_4ch = np.expand_dims(png_4ch, axis=0).astype(np.float32) / 255;

        splitr = np.random.random();
        if splitr < 0.2:
            X_test = np.concatenate((X_test, png_4ch));
            y_test = np.concatenate((y_test, np.expand_dims(np.array(lab), axis = 0)));
        elif splitr >= 0.2 and splitr < 0.25:
            X_val = np.concatenate((X_val, png_4ch));
            y_val = np.concatenate((y_val, np.expand_dims(np.array(lab), axis = 0)));
        elif splitr >= 0.25:
            X_train = np.concatenate((X_train, png_4ch));
            y_train = np.concatenate((y_train, np.expand_dims(np.array(lab), axis = 0)));

    X_train = (X_train.astype(np.float32) - mu) / sigma;
    X_val = (X_val.astype(np.float32) - mu) / sigma;
    X_test = (X_test.astype(np.float32) - mu) / sigma;
    y_train = y_train.astype(np.float32);
    y_val = y_val.astype(np.float32);
    y_test = y_test.astype(np.float32);

    print "Data Loaded", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape;
    return X_train, y_train, X_val, y_val, X_test, y_test;


def val_fn_epoch(classn, val_fn, X_val, a_val, y_val):
    val_err = 0;
    Er = np.empty(shape = (0, 100), dtype = np.float32);
    Or = np.empty(shape = (0, classn), dtype = np.float32);
    Tr = np.empty(shape = (0, classn), dtype = np.float32);
    val_batches = 0;
    for batch in iterate_minibatches(X_val, a_val, y_val, batchsize = 100, shuffle = False):
        inputs, augs, targets = batch;
        err, encode, hidden, smth_act, output = val_fn(inputs, augs, targets);
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


def train_round(train_fn, val_fn, classn, X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, y_test):
    print("Starting training...");
    print("TrLoss\t\tVaLoss\t\tVaRMSE\t\tVaPear\t\tEpochs\t\tTime");
    num_epochs = 3000;
    batchsize = 100;
    for epoch in range(num_epochs):
        train_err = 0;
        train_batches = 0;
        start_time = time.time();
        for batch in iterate_minibatches(X_train, a_train, y_train, batchsize, shuffle = True):
            inputs, augs, targets = batch;
            inputs = data_aug(inputs);
            train_err += train_fn(inputs, augs, targets);
            train_batches += 1;
        train_err = train_err / train_batches;

        if epoch % 100 == 0:
            # And a full pass over the validation data:
            val_err, Er, Or, Tr = val_fn_epoch(classn, val_fn, X_val, a_val, y_val);
            # Then we print the results for this epoch:
            print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}\t\t{:.3f}".format(
                train_err, val_err, comp_rmse(Or, Tr), comp_pear(Or, Tr), epoch + 1, num_epochs, time.time() - start_time));

    # Return a new set of features.
    _, _, train_Or, _ = val_fn_epoch(classn, val_fn, X_train, a_train, y_train);
    _, _, val_Or, _ = val_fn_epoch(classn, val_fn, X_val, a_val, y_val);
    _, _, test_Or, _ = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    return train_Or, val_Or, test_Or;


def stack_pretrain_round(stack_train_fn, val_fn, classn, X_train, a_train, y_train, X_val, a_val, y_val):
    print("Starting stack_pretraining...");
    print("TrLoss\t\tVaLoss\t\tVaRMSE\t\tVaPear\t\tEpochs\t\tTime");
    stack_num_epochs = 500;
    stack_batchsize = 100;
    for epoch in range(stack_num_epochs):
        train_err = 0;
        train_batches = 0;
        start_time = time.time();
        for batch in iterate_minibatches(X_train, a_train, y_train, stack_batchsize, shuffle = True):
            inputs, augs, targets = batch;
            inputs = data_aug(inputs);
            train_err += stack_train_fn(inputs, augs, targets);
            train_batches += 1;
        train_err = train_err / train_batches;

        if epoch % 100 == 0:
            # And a full pass over the validation data:
            val_err, Er, Or, Tr = val_fn_epoch(classn, val_fn, X_val, a_val, y_val);
            # Then we print the results for this epoch:
            print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}\t\t{:.3f}".format(
                train_err, val_err, comp_rmse(Or, Tr), comp_pear(Or, Tr), epoch + 1, stack_num_epochs, time.time() - start_time));


def build_network_from_ae(classn):
    input_var = T.tensor4('inputs');
    aug_var = T.matrix('aug_var');
    target_var = T.matrix('targets');

    ae = pickle.load(open('model_4ch/conv_ae.pkl', 'rb'));

    input_layer_index = map(lambda pair : pair[0], ae.layers).index('input');
    first_layer = ae.get_all_layers()[input_layer_index + 1];
    input_layer = layers.InputLayer(shape = (None, 4, 32, 32), input_var = input_var);
    first_layer.input_layer = input_layer;

    encode_layer_index = map(lambda pair : pair[0], ae.layers).index('encode_layer');
    encode_layer = ae.get_all_layers()[encode_layer_index];
    aug_layer = layers.InputLayer(shape=(None, classn), input_var = aug_var);

    cat_layer = lasagne.layers.ConcatLayer([encode_layer, aug_layer], axis = 1);

    # conventional recitified linear units
    #hidden_layer = layers.DenseLayer(incoming = cat_layer, num_units = 200, nonlinearity = rectify);
    #network = layers.DenseLayer(incoming = hidden_layer, num_units = classn, nonlinearity = linear);
    #stack_params = [network.W, network.b, hidden_layer.W, hidden_layer.b];

    # smooth activation function
    hidden_layer = layers.DenseLayer(incoming = cat_layer, num_units = 200, nonlinearity = linear);
    ago_layer = AgoLayer(incoming = hidden_layer, num_segs = 20);
    network = layers.DenseLayer(incoming = ago_layer, num_units = classn, nonlinearity = linear);
    stack_params = [network.W, network.b, hidden_layer.W, hidden_layer.b, ago_layer.W];

    return (encode_layer, hidden_layer, ago_layer, network), input_var, aug_var, target_var, stack_params;


def make_training_functions(network_layers, input_var, aug_var, target_var, stack_params, weight_decay):
    encode_layer, hidden_layer, ago_layer, network = network_layers;

    output = lasagne.layers.get_output(network, deterministic = True);
    loss = lasagne.objectives.squared_error(output, target_var).mean() + \
           weight_decay * regularization.regularize_network_params(
                   layer = network, penalty = regularization.l2, tags={'regularizable' : True});

    params = layers.get_all_params(network, trainable = True);
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = 0.00002, momentum = 0.95);
    stack_updates = lasagne.updates.nesterov_momentum(loss, stack_params, learning_rate = 0.00001, momentum = 0.95);

    encode = lasagne.layers.get_output(encode_layer, deterministic = True);
    hidden = lasagne.layers.get_output(hidden_layer, deterministic = True);
    smth_act = lasagne.layers.get_output(ago_layer, deterministic = True);

    val_fn = theano.function([input_var, aug_var, target_var], [loss, encode, hidden, smth_act, output]);
    train_fn = theano.function([input_var, aug_var, target_var], loss, updates = updates);
    stack_train_fn = theano.function([input_var, aug_var, target_var], loss, updates = stack_updates);

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
    mu = pickle.load(open('model_4ch/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model_4ch/conv_sigma.pkl', 'rb'));

    if not os.path.isfile('./visual/truth.txt'):
        with open('./visual/truth.txt', 'w'):
            pass;

    ind = sum(1 for line in open('./visual/truth.txt'));
    for xi in range(X.shape[0]):
        ind += 1;
        im = ((X[xi].transpose() * sigma + mu) * 255).astype(np.uint8);
        misc.imsave("./visual/case_{}.png".format(ind), im);

    f = file('./visual/output.txt', 'a');
    np.savetxt(f, Or, fmt='%.4f');
    f.close();

    f = file('./visual/truth.txt', 'a');
    np.savetxt(f, Tr, fmt='%.4f');
    f.close();


def split_validation(classn, weight_decay):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(classn);

    network_layers, input_var, aug_var, target_var, stack_params = build_network_from_ae(classn);
    val_fn, train_fn, stack_train_fn = make_training_functions(network_layers, input_var, aug_var, target_var, stack_params, weight_decay);

    a_train = np.zeros((X_train.shape[0], classn), dtype = np.float32);
    a_val = np.zeros((X_val.shape[0], classn), dtype = np.float32);
    a_test = np.zeros((X_test.shape[0], classn), dtype = np.float32);

    stack_pretrain_round(stack_train_fn, val_fn, classn, X_train, a_train, y_train, X_val, a_val, y_val);

    n_round = 1;
    for train_i in range(n_round - 1):
        print("Round {}".format(train_i));
        a_train, a_val, a_test = train_round(train_fn, val_fn, classn,
            X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, y_test);

    print("Round {}".format(n_round - 1));
    train_round(train_fn, val_fn, classn,
            X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, y_test);

    # Testing
    _, Er, Or, Tr = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    save_visual_cases(X_test, Or, Tr);
    return Er, Or, Tr;


def main():
    classn = 1;
    split_n = 10;
    weight_decay = 0.00002;
    sys.setrecursionlimit(10000);

    for v in range(split_n):
        Er, Or, Tr = split_validation(classn, weight_decay);
        save_result("regression_sp_{}".format(v), Er, Or, Tr);
        print("[CNN] RMSE: {:.4f} Pear: {:.4f}".format(comp_rmse(Or, Tr), comp_pear(Or, Tr)));


if __name__ == "__main__":
    main();

