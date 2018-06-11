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
from extensive_data_aug_60x60_rand_deter import data_aug
from ch_inner_prod import ChInnerProd, ChInnerProdMerge

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

APS = 300;
PS = 60;
cv_v = int(sys.argv[1]);
LearningRate = theano.shared(np.array(5e-3, dtype=np.float32));

filename_code = 21;
filesave_code = 6;
filename_model_ae = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_model_{}.pkl'.format(filename_code, filesave_code);
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_mu.pkl'.format(filename_code);
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_sigma.pkl'.format(filename_code);

mu = pickle.load(open(filename_mu, 'rb'));
sigma = pickle.load(open(filename_sigma, 'rb'));
model_dump = 'model_vals/' + sys.argv[0];
aug_fea_n = 96;

def iterate_minibatches(inputs, augs, targets, batchsize, shuffle=False):
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
            excerpt = indices[start_idx:start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt], augs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx + batchsize : len(inputs)];
        else:
            excerpt = slice(start_idx + batchsize, len(inputs));
        yield inputs[excerpt], augs[excerpt], targets[excerpt];


def load_data_folder(classn, folder, is_train):
    X = np.zeros(shape=(100000, 3, APS, APS), dtype=np.float32);
    A = np.zeros(shape=(100000, aug_fea_n), dtype=np.float32);
    y = np.zeros(shape=(100000, classn), dtype=np.int32);

    nline = 0;
    lines = [line.rstrip('\n') for line in open(folder + '/label.txt')];
    for line in lines:
        img = line.split()[0];
        lab = np.array([int(int(line.split()[1]) > 0)]);
        aug = np.array([float(x) for x in line.split()[5].split(',')], dtype=np.float32);

        png = np.array(Image.open(folder + '/' + img).convert('RGB')).transpose();
        if (png.shape[1] == 500):
            png = png[:, 100:-100, 100:-100];
        X[nline], A[nline], y[nline] = png, aug, lab;
        nline += 1;

    X = X[0:nline];
    A = A[0:nline];
    y = y[0:nline];
    return X, A, y;


def load_data_split(classn, folders, is_train):
    X = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    A = np.zeros(shape=(0, aug_fea_n), dtype=np.float32);
    y = np.zeros(shape=(0, classn), dtype=np.int32);
    for folder in folders:
        X_split, A_split, y_split = load_data_folder(classn, folder, is_train);
        X = np.concatenate((X, X_split));
        A = np.concatenate((A, A_split));
        y = np.concatenate((y, y_split));
    return X, A, y;


def load_data(classn, valid_num):
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    A_train = np.zeros(shape=(0, aug_fea_n), dtype=np.float32);
    y_train = np.zeros(shape=(0, classn), dtype=np.int32);

    all_data_folder = './data/vals/';
    lines = [line.rstrip('\n') for line in open(all_data_folder + '/nucleus_level_lym_click.txt')];
    valid_i = 0;
    for line in lines:
        split_folders = [all_data_folder + s for s in line.split()];
        if valid_i == valid_num:
            X_test, A_test, y_test = load_data_split(classn, split_folders, False);
        else:
            X_split, A_split, y_split = load_data_split(classn, split_folders, True);
            X_train = np.concatenate((X_train, X_split));
            A_train = np.concatenate((A_train, A_split));
            y_train = np.concatenate((y_train, y_split));
        valid_i += 1;

    print "Data Loaded", X_train.shape, A_train.shape, y_train.shape, X_test.shape, A_test.shape, y_test.shape;
    return X_train, A_train, y_train, X_test, A_test, y_test;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def multi_win_during_val(val_fn, inputs, augs, targets):
    for idraw in [100, 110, 120, 130, 140]:
        for jdraw in [100, 110, 120, 130, 140]:
            inpt_multiwin = data_aug(inputs, mu, sigma, deterministic=True, idraw=idraw, jdraw=jdraw);
            err_pat, output_pat = val_fn(inpt_multiwin, augs, targets);
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


def val_fn_epoch(classn, val_fn, X_val, a_val, y_val):
    val_err = 0;
    Pr = np.empty(shape = (100000, classn), dtype = np.int32);
    Or = np.empty(shape = (100000, classn), dtype = np.float32);
    Tr = np.empty(shape = (100000, classn), dtype = np.int32);
    val_batches = 0;
    nline = 0;
    for batch in iterate_minibatches(X_val, a_val, y_val, batchsize = 100, shuffle = False):
        inputs, augs, targets = batch;
        err, output = multi_win_during_val(val_fn, inputs, augs, targets);
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


def confusion_matrix(Pr, Tr):
    tpos = np.sum((Pr==1) * (Tr==1));
    tneg = np.sum((Pr==0) * (Tr==0));
    fpos = np.sum((Pr==1) * (Tr==0));
    fneg = np.sum((Pr==0) * (Tr==1));
    return tpos, tneg, fpos, fneg;


def auc_roc(Pr, Tr):
    fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0);
    return auc(fpr, tpr);


def train_round(num_epochs, network, valid_num, train_fn, val_fn, classn, X_train, a_train, y_train, X_test, a_test, y_test):
    print("Starting training...");
    print("TrLoss\t\tVaLoss\t\tVaHamm\t\tAUC\t\tCMatrix\t\t\tEpochs\t\tTime");
    batchsize = 100;
    start_time = time.time();
    for epoch in range(num_epochs+1):
        train_err = 0;
        train_batches = 0;
        for batch in iterate_minibatches(X_train, a_train, y_train, batchsize, shuffle = True):
            inputs, augs, targets = batch;
            inputs = data_aug(inputs, mu, sigma);
            train_err += train_fn(inputs, augs, targets);
            train_batches += 1;
        train_err = train_err / train_batches;

        if epoch % 1 == 0:
            # And a full pass over the validation data:
            test_err, test_ham, _, Pr, Or, Tr = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
            tpos, tneg, fpos, fneg = confusion_matrix(Pr, Tr);
            val_auc = auc_roc(Or, Tr);
            # Then we print the results for this epoch:
            print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}/{}/{}\t\t{}/{}\t\t{:.3f}".format(
                train_err, test_err, test_ham, val_auc, tpos, tneg, fpos, fneg, epoch + 1, num_epochs, time.time() - start_time));
            start_time = time.time();

        if epoch % 10 == 0:
            param_values = layers.get_all_param_values(network);
            pickle.dump(param_values, open(model_dump + "_e{}_cv{}.pkl".format(epoch, valid_num), 'w'));

        if epoch == 6:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 10:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));

    # Return a new set of features.
    _, _, _, _, train_Or, _ = val_fn_epoch(classn, val_fn, X_train, a_train, y_train);
    _, _, _, _, test_Or, _ = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    return train_Or, test_Or;


def build_network_from_ae(classn):
    input_var = T.tensor4('input_var');

    layer = layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);
    layer = batch_norm(layers.Conv2DLayer(layer, 100,  filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 120,  filter_size=(5,5), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = layers.Pool2DLayer(layer, pool_size=(2,2), stride=2, mode='average_inc_pad');
    layer = batch_norm(layers.Conv2DLayer(layer, 240,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Conv2DLayer(layer, 320,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    layer = layers.Pool2DLayer(layer, pool_size=(2,2), stride=2, mode='average_inc_pad');
    layer = batch_norm(layers.Conv2DLayer(layer, 640,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    prely = batch_norm(layers.Conv2DLayer(layer, 1024, filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));

    featm    = batch_norm(layers.Conv2DLayer(prely, 640, filter_size=(1,1), nonlinearity=leaky_rectify));
    feat_map = batch_norm(layers.Conv2DLayer(featm, 100, filter_size=(1,1), nonlinearity=rectify, name="feat_map"));
    maskm    = batch_norm(layers.Conv2DLayer(prely, 100, filter_size=(1,1), nonlinearity=leaky_rectify));
    mask_rep = batch_norm(layers.Conv2DLayer(maskm, 1,   filter_size=(1,1), nonlinearity=None),   beta=None, gamma=None);
    mask_map = SoftThresPerc(mask_rep, perc=0.0, alpha=0.1, beta=init.Constant(0.5), tight=100.0, bias=-10, name="mask_map");
    enlyr    = ChInnerProdMerge(feat_map, mask_map, name="encoder");

    layer = batch_norm(layers.Deconv2DLayer(enlyr, 1024, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 640,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 640,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 320,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 320,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 240,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 120,  filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer = batch_norm(layers.Deconv2DLayer(layer, 100,  filter_size=(5,5), stride=1, crop='same', nonlinearity=leaky_rectify));
    layer =            layers.Deconv2DLayer(layer, 3,    filter_size=(1,1), stride=1, crop='same', nonlinearity=identity);

    glblf = batch_norm(layers.Conv2DLayer(prely, 128,  filter_size=(1,1), nonlinearity=leaky_rectify));
    glblf = layers.Pool2DLayer(glblf, pool_size=(5,5), stride=5, mode='average_inc_pad');
    glblf = batch_norm(layers.Conv2DLayer(glblf, 64,   filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    gllyr = batch_norm(layers.Conv2DLayer(glblf, 5,    filter_size=(1,1), nonlinearity=rectify), name="global_feature");

    glblf = batch_norm(layers.Deconv2DLayer(gllyr, 256, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(9,9), stride=5, crop=(2,2),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 64,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32,  filter_size=(4,4), stride=2, crop=(1,1),  nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf = batch_norm(layers.Deconv2DLayer(glblf, 32,  filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    glblf =            layers.Deconv2DLayer(glblf, 3,   filter_size=(1,1), stride=1, crop='same', nonlinearity=identity);

    layer = layers.ElemwiseSumLayer([layer, glblf]);

    network = ReshapeLayer(layer, ([0], -1));
    layers.set_all_param_values(network, pickle.load(open(filename_model_ae, 'rb')));
    mask_map.beta.set_value(np.float32(-10.0*mask_map.beta.get_value()));

    # Adding more layers
    aug_var = T.matrix('aug_var');
    target_var = T.imatrix('targets');
    add_a = layers.Conv2DLayer(enlyr, 320, filter_size=(1,1), nonlinearity=leaky_rectify);
    add_b = layers.Conv2DLayer(add_a, 320, filter_size=(1,1), nonlinearity=leaky_rectify);
    add_c = layers.Conv2DLayer(add_b, 320, filter_size=(1,1), nonlinearity=leaky_rectify);
    add_d = layers.Conv2DLayer(add_c, 320, filter_size=(1,1), nonlinearity=leaky_rectify);
    add_0 = layers.Pool2DLayer(add_d, pool_size=(15,15), stride=15, mode='average_inc_pad');
    add_1 = layers.DenseLayer(add_0, 100, nonlinearity=leaky_rectify);

    add_2 = layers.DenseLayer(gllyr, 320, nonlinearity=leaky_rectify);
    add_3 = layers.DenseLayer(add_2, 320, nonlinearity=leaky_rectify);
    add_4 = layers.DenseLayer(add_3, 100, nonlinearity=leaky_rectify);

    aug_layer = layers.InputLayer(shape=(None, aug_fea_n), input_var=aug_var);

    cat_layer = lasagne.layers.ConcatLayer([add_1, add_4, aug_layer], axis=1);

    hidden_layer = layers.DenseLayer(cat_layer, 80, nonlinearity=leaky_rectify);
    network = layers.DenseLayer(hidden_layer, classn, nonlinearity=sigmoid);

    new_params = [add_a.W, add_a.b, add_b.W, add_b.b, add_c.W, add_c.b, add_d.W, add_d.b, add_1.W, add_1.b, add_2.W, add_2.b, add_3.W, add_3.b, add_4.W, add_4.b, hidden_layer.W, hidden_layer.b, network.W, network.b];

    return network, new_params, input_var, aug_var, target_var;

def make_training_functions(network, new_params, input_var, aug_var, target_var):
    output = lasagne.layers.get_output(network, deterministic=True, batch_norm_use_averages=True, batch_norm_update_averages=False);
    loss = lasagne.objectives.binary_crossentropy(output, target_var).mean();

    deter_output = lasagne.layers.get_output(network, deterministic=True);
    deter_loss = lasagne.objectives.binary_crossentropy(deter_output, target_var).mean();

    params = layers.get_all_params(network, trainable=True);
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=LearningRate, momentum=0.985);
    new_params_updates = lasagne.updates.nesterov_momentum(loss, new_params, learning_rate=LearningRate, momentum=0.985);

    val_fn = theano.function([input_var, aug_var, target_var], [deter_loss, deter_output]);
    train_fn = theano.function([input_var, aug_var, target_var], loss, updates=updates);
    new_params_train_fn = theano.function([input_var, aug_var, target_var], loss, updates=new_params_updates);

    return train_fn, new_params_train_fn, val_fn;


def get_aug_feas(X):
    aug_feas = np.zeros((X.shape[0], aug_fea_n), dtype=np.float32);
    return aug_feas;


def split_validation(classn, valid_num):
    X_train, A_train, y_train, X_test, A_test, y_test = load_data(classn, valid_num);

    network, new_params, input_var, aug_var, target_var = build_network_from_ae(classn);
    train_fn, new_params_train_fn, val_fn = make_training_functions(network, new_params, input_var, aug_var, target_var);

    a_train = get_aug_feas(X_train);
    a_test = get_aug_feas(X_test);
    train_round(15,  network, valid_num, new_params_train_fn, val_fn, classn, X_train, a_train, y_train, X_test, a_test, y_test);

    # Testing
    _, _, _, Pr, Or, Tr = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    return Pr, Or, Tr;


def main():
    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    print "DOING CV", cv_v;
    Pr, Or, Tr = split_validation(classn, cv_v);
    print("[CNN] Hamming: {:.4f}\tAccuracy: {:.4f}".format(1 - hamming_loss(Tr, Pr), accuracy_score(Tr, Pr)));


if __name__ == "__main__":
    main();

