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
from lasagne.objectives import mse
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import softmax
from scipy import misc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score

from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets);
    if shuffle:
        indices = np.arange(len(inputs));
        np.random.shuffle(indices);
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx:len(inputs)];
        else:
            excerpt = slice(start_idx, len(inputs));
        yield inputs[excerpt], targets[excerpt];

def data_aug(X):
    bs = X.shape[0];
    h_indices = np.random.choice(bs, bs / 2, replace=False);  # horizontal flip
    v_indices = np.random.choice(bs, bs / 2, replace=False);  # vertical flip
    r_indices = np.random.choice(bs, bs / 2, replace=False);  # 90 degree rotation

    X[h_indices] = X[h_indices, :, :, ::-1];
    X[v_indices] = X[v_indices, :, ::-1, :];
    X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3);

    return X;


def load_data(mu, sigma, classn):
    X_test = np.empty(shape=(0, 3, 32, 32));
    X_val = np.empty(shape=(0, 3, 32, 32));
    X_train = np.empty(shape=(0, 3, 32, 32));

    y_test = np.empty(shape=(0, classn));
    y_val = np.empty(shape=(0, classn));
    y_train = np.empty(shape=(0, classn));

    lines = [line.rstrip('\n') for line in open('./data/image/label.txt')];
    linen = 0;
    for line in lines:
        linen += 1;
        img = line.split('\t')[0];
        lab = [int(x) for x in line.split('\t')[1].split()];
        png = misc.imread('./data/' + img).transpose()[0 : 3, 9 : 41, 9 : 41];
        png = np.expand_dims(png, axis=0).astype(np.float32) / 255;
        if linen % 100 <= 19:
            X_test = np.concatenate((X_test, png));
            y_test = np.concatenate((y_test, np.expand_dims(np.array(lab), axis=0)));
        elif linen % 100 >= 20 and linen % 100 <= 24:
            X_val = np.concatenate((X_val, png));
            y_val = np.concatenate((y_val, np.expand_dims(np.array(lab), axis=0)));
        elif linen % 100 >= 25:
            X_train = np.concatenate((X_train, png));
            y_train = np.concatenate((y_train, np.expand_dims(np.array(lab), axis=0)));

    X_train = X_train.astype(np.float32);
    X_val = X_val.astype(np.float32);
    X_test = X_test.astype(np.float32);
    y_train = y_train.astype(np.uint8);
    y_val = y_val.astype(np.uint8);
    y_test = y_test.astype(np.uint8);

    X_train = (X_train - mu) / sigma;
    X_val = (X_val - mu) / sigma;
    X_test = (X_test - mu) / sigma;

    print "Data Loaded", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape;
    sys.stdout.flush();
    return X_train, y_train, X_val, y_val, X_test, y_test;


classn = 19;
sys.setrecursionlimit(10000);
ae = pickle.load(open('model/conv_ae.pkl','rb'));
mu = pickle.load(open('model/conv_mu.pkl','rb'));
sigma = pickle.load(open('model/conv_sigma.pkl','rb'));

X_train, y_train, X_val, y_val, X_test, y_test = load_data(mu, sigma, classn);
input_var = T.tensor4('inputs');
target_var = T.imatrix('targets');

input_layer_index = map(lambda pair : pair[0], ae.layers).index('input');
first_layer = ae.get_all_layers()[input_layer_index + 1];
input_layer = layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var);
first_layer.input_layer = input_layer;

encode_layer_index = map(lambda pair : pair[0], ae.layers).index('encode_layer');
encode_layer = ae.get_all_layers()[encode_layer_index];
fc_layer = layers.DenseLayer(incoming = encode_layer, num_units = 30, nonlinearity = rectify);
network = layers.DenseLayer(incoming = fc_layer, num_units = classn, nonlinearity = sigmoid);

prediction = layers.get_output(network);
loss = lasagne.objectives.binary_crossentropy(prediction, target_var).mean();

params = layers.get_all_params(network, trainable=True);
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0005, momentum=0.975);

test_output = lasagne.layers.get_output(network, deterministic=True);
test_loss = lasagne.objectives.binary_crossentropy(test_output, target_var).mean();

test_acc = T.mean(T.eq(T.gt(test_output, 0.5), target_var), dtype=theano.config.floatX);
test_pred = T.gt(test_output, 0.5);

train_fn = theano.function([input_var, target_var], loss, updates=updates);
val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_pred, test_output]);

print("Starting training...");
print("TrLoss\t\tVaLoss\t\tVaAcc\t\tEpochs\t\tTime");
sys.stdout.flush();
num_epochs = 300;
batchsize = 100;
for epoch in range(num_epochs):
    train_err = 0;
    train_batches = 0;
    start_time = time.time();
    for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
        inputs, targets = batch;
        inputs = data_aug(inputs);
        train_err += train_fn(inputs, targets);
        train_batches += 1;

    # And a full pass over the validation data:
    val_err = 0;
    val_acc = 0;
    val_batches = 0;
    for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
        inputs, targets = batch;
        err, acc, pred, output = val_fn(inputs, targets);
        val_err += err;
        val_acc += acc;
        val_batches += 1;

    # Then we print the results for this epoch:
    print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}\t\t{:.3f}".format(
            train_err / train_batches, val_err / val_batches, val_acc / val_batches,
            epoch + 1, num_epochs, time.time() - start_time));
    sys.stdout.flush();

# After training, we compute and print the test error:
test_err = 0;
test_acc = 0;
test_shape_acc = 0;
test_batches = 0;
P = np.empty(shape=(0, classn));
T = np.empty(shape=(0, classn));
for batch in iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
    inputs, targets = batch;
    err, acc, pred, output = val_fn(inputs, targets);
    P = np.concatenate((P, np.array(pred)));
    T = np.concatenate((T, np.array(targets)));
    test_err += err;
    test_acc += acc;
    test_shape_acc += ((output[:, 12:].argmax(axis = 1) == targets[:, 12:].argmax(axis = 1)).sum()
            / float(targets.shape[0]));
    test_batches += 1;
print("Final results:");
print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches));
print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100));
print("  test shape accuracy:\t\t{:.2f} %".format(test_shape_acc / test_batches * 100));
sys.stdout.flush();

print("Hit matrix:");
hit = np.zeros(shape=(classn, 2), dtype=np.uint);
for i in range(T.shape[0]):
    for truth_ind in range(T.shape[1]):
        if T[i][truth_ind] == 1:
            hit[truth_ind, P[i][truth_ind]] += 1;
print '\n'.join('\t'.join(str(cell) for cell in row) for row in hit);
sys.stdout.flush();

print("Mis matrix:");
mis = np.zeros(shape=(classn, 2), dtype=np.uint);
for i in range(T.shape[0]):
    for truth_ind in range(T.shape[1]):
        if T[i][truth_ind] == 0:
            mis[truth_ind, P[i][truth_ind]] += 1;
print '\n'.join('\t'.join(str(cell) for cell in row) for row in mis);
sys.stdout.flush();

