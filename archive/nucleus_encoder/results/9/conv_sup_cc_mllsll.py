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
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import sigmoid
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import softmax
from scipy import misc

from shape import ReshapeLayer
from unpool import Unpool2DLayer
from flipiter import FlipBatchIterator

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
    h_indices = np.random.choice(bs, bs / 2, replace=False);  # horizontal flip
    v_indices = np.random.choice(bs, bs / 2, replace=False);  # vertical flip
    r_indices = np.random.choice(bs, bs / 2, replace=False);  # 90 degree rotation

    X[h_indices] = X[h_indices, :, :, ::-1];
    X[v_indices] = X[v_indices, :, ::-1, :];
    for rot in range(np.random.randint(3) + 1):
        X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3);

    return X;


def load_data(classn):
    mu = pickle.load(open('model/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model/conv_sigma.pkl', 'rb'));

    X_test = np.empty(shape = (0, 3, 32, 32));
    X_val = np.empty(shape = (0, 3, 32, 32));
    X_train = np.empty(shape = (0, 3, 32, 32));

    y_test = np.empty(shape = (0, classn));
    y_val = np.empty(shape = (0, classn));
    y_train = np.empty(shape = (0, classn));

    lines = [line.rstrip('\n') for line in open('./data/image/label.txt')];
    for line in lines:
        img = line.split('\t')[0];
        lab = [int(x) for x in line.split('\t')[1].split()];
        png = misc.imread('./data/' + img).transpose()[0 : 3, 9 : 41, 9 : 41];
        png = np.expand_dims(png, axis=0).astype(np.float32) / 255;
        splitr = np.random.random();
        if splitr < 0.2:
            X_test = np.concatenate((X_test, png));
            y_test = np.concatenate((y_test, np.expand_dims(np.array(lab), axis=0)));
        elif splitr >= 0.2 and splitr < 0.25:
            X_val = np.concatenate((X_val, png));
            y_val = np.concatenate((y_val, np.expand_dims(np.array(lab), axis=0)));
        elif splitr >= 0.25:
            X_train = np.concatenate((X_train, png));
            y_train = np.concatenate((y_train, np.expand_dims(np.array(lab), axis=0)));

    X_train = (X_train.astype(np.float32) - mu) / sigma;
    X_val = (X_val.astype(np.float32) - mu) / sigma;
    X_test = (X_test.astype(np.float32) - mu) / sigma;
    y_train = y_train.astype(np.float32);
    y_val = y_val.astype(np.float32);
    y_test = y_test.astype(np.float32);

    print "Data Loaded", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape;
    return X_train, y_train, X_val, y_val, X_test, y_test;


def from_output_to_pred(output):
    pred = np.copy(output);
    for i in range(pred.shape[0]):
        pred[i, 12:] = (pred[i, 12:] == np.max(pred[i, 12:])).astype(np.float32)
    pred = (pred >= 0.5).astype(np.float32);
    return pred;


def val_fn_epoch(classn, val_fn, X_val, a_val, y_val):
    val_err = 0;
    Er = np.empty(shape = (0, 100), dtype = np.float32);
    Pr = np.empty(shape = (0, classn), dtype = np.float32);
    Or = np.empty(shape = (0, classn), dtype = np.float32);
    Tr = np.empty(shape = (0, classn), dtype = np.float32);
    val_batches = 0;
    for batch in iterate_minibatches(X_val, a_val, y_val, batchsize = 100, shuffle = False):
        inputs, augs, targets = batch;
        err, encode, output = val_fn(inputs, augs, targets);
        pred = from_output_to_pred(output);
        val_err += err;
        Er = np.concatenate((Er, encode));
        Pr = np.concatenate((Pr, pred));
        Or = np.concatenate((Or, output));
        Tr = np.concatenate((Tr, targets));
        val_batches += 1;
    val_err = val_err / val_batches;
    val_ham = (1 - hamming_loss(Tr, Pr));
    val_acc = accuracy_score(Tr, Pr);
    return val_err, val_ham, val_acc, Er, Pr, Or, Tr;

def train_round(train_fn, val_fn, classn, X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, y_test):
    print("Starting training...");
    print("TrLoss\t\tVaLoss\t\tVaHamm\t\tEpochs\t\tTime");
    num_epochs = 500;
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
            val_err, val_ham, _, _, _, _, _ = val_fn_epoch(classn, val_fn, X_val, a_val, y_val);
            # Then we print the results for this epoch:
            print("{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{}/{}\t\t{:.3f}".format(
                train_err, val_err, val_ham, epoch + 1, num_epochs, time.time() - start_time));

    # Return a new set of features.
    _, _, _, _, _, train_Or, _ = val_fn_epoch(classn, val_fn, X_train, a_train, y_train);
    _, _, _, _, _, val_Or, _ = val_fn_epoch(classn, val_fn, X_val, a_val, y_val);
    _, _, _, _, _, test_Or, _ = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    return train_Or, val_Or, test_Or;


def build_network_from_ae(classn):
    input_var = T.tensor4('inputs');
    aug_var = T.matrix('aug_var');
    target_var = T.matrix('targets');

    ae = pickle.load(open('model/conv_ae.pkl', 'rb'));

    input_layer_index = map(lambda pair : pair[0], ae.layers).index('input');
    first_layer = ae.get_all_layers()[input_layer_index + 1];
    input_layer = layers.InputLayer(shape=(None, 3, 32, 32), input_var = input_var);
    first_layer.input_layer = input_layer;

    encode_layer_index = map(lambda pair : pair[0], ae.layers).index('encode_layer');
    encode_layer = ae.get_all_layers()[encode_layer_index];
    aug_layer = layers.InputLayer(shape=(None, classn), input_var = aug_var);

    cat_layer = lasagne.layers.ConcatLayer([encode_layer, aug_layer], axis = 1);
    hidden_layer = layers.DenseLayer(incoming = cat_layer, num_units = 100, nonlinearity = rectify);

    network_mll = layers.DenseLayer(incoming = hidden_layer, num_units = 12, nonlinearity = sigmoid);
    network_sll = layers.DenseLayer(incoming = hidden_layer, num_units = 7, nonlinearity = sigmoid);
    network = lasagne.layers.ConcatLayer([network_mll, network_sll], axis = 1);

    return network, encode_layer, input_var, aug_var, target_var;

def make_training_functions(network, encode_layer, input_var, aug_var, target_var):
    output = lasagne.layers.get_output(network, deterministic=True);
    loss = lasagne.objectives.binary_crossentropy(output[:, :12], target_var[:, :12]).mean() + \
           lasagne.objectives.binary_crossentropy(output[:, 12:], target_var[:, 12:]).mean();

    params = layers.get_all_params(network, trainable=True);
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.0002, momentum=0.975);

    encode = lasagne.layers.get_output(encode_layer, deterministic=True);

    val_fn = theano.function([input_var, aug_var, target_var], [loss, encode, output]);
    train_fn = theano.function([input_var, aug_var, target_var], loss, updates=updates);

    return train_fn, val_fn;


def hit_mis(classn, Pr, Tr):
    print("Hit matrix:");
    hit = np.zeros(shape=(classn, 2), dtype=np.uint);
    for i in range(Tr.shape[0]):
        for truth_ind in range(Tr.shape[1]):
            if Tr[i][truth_ind] == 1:
                hit[truth_ind, Pr[i][truth_ind]] += 1;
    print '\n'.join('\t'.join(str(cell) for cell in row) for row in hit);

    print("Mis matrix:");
    mis = np.zeros(shape=(classn, 2), dtype=np.uint);
    for i in range(Tr.shape[0]):
        for truth_ind in range(Tr.shape[1]):
            if Tr[i][truth_ind] == 0:
                mis[truth_ind, Pr[i][truth_ind]] += 1;
    print '\n'.join('\t'.join(str(cell) for cell in row) for row in mis);


def svm_br(classn, X_train, a_train, y_train, X_test, a_test, y_test):
    _, _, _, Er, _, _, Tr = val_fn_epoch(classn, val_fn, X_train, a_train, y_train);
    model = OneVsRestClassifier(SVC(kernel = 'rbf')).fit(Er, Tr);
    _, _, _, Er, _, _, Tr = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    Pr = model.predict(Er);
    print("[SVM] Hamming: {:.4f}\tAccuracy: {:.4f}".format(1 - hamming_loss(Tr, Pr), accuracy_score(Tr, Pr)));


def draw_roc(plot_str, classes, Tr, Or):
    auc_vec = np.zeros(shape = len(classes));

    plt.figure(figsize = (13, 10));
    for c in range(Tr.shape[1]):
        if np.sum(Tr[:, c]) != 0:
            fpr, tpr, thresholds = roc_curve(Tr[:, c], Or[:, c], pos_label = 1);
            label_str = classes[c] + \
                " AUC[{:.4f}] Pos[{}] Neg[{}]".format(auc(fpr, tpr), np.sum(Tr[:, c] == 1), np.sum(Tr[:, c] == 0));
            auc_vec[c] = auc(fpr, tpr);
            plt.plot(fpr, tpr, label = label_str);
            plt.legend(loc = 'best')
        if c == 11:
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.title('Attribute Classification ROC');
            plt.savefig('curves/' + plot_str + '_attri.png');
            plt.clf();
        if c == 18:
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate');
            plt.title('Shape Classification ROC');
            plt.savefig('curves/' + plot_str + '_shape.png');
            plt.clf();
    return auc_vec;


def save_result(cache_str, Er, Pr, Or, Tr):
    cache_dir = './cache/' + cache_str + '/';
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir);
    np.save(cache_dir + 'Er', Er);
    np.save(cache_dir + 'Pr', Pr);
    np.save(cache_dir + 'Or', Or);
    np.save(cache_dir + 'Tr', Tr);


def load_result(cache_str):
    cache_dir = './cache/' + cache_str + '/';
    Er = np.load(cache_dir + 'Er.npy');
    Pr = np.load(cache_dir + 'Pr.npy');
    Or = np.load(cache_dir + 'Or.npy');
    Tr = np.load(cache_dir + 'Tr.npy');
    return Er, Pr, Or, Tr;


def save_visual_cases(X, Or, Tr):
    mu = pickle.load(open('model/conv_mu.pkl', 'rb'));
    sigma = pickle.load(open('model/conv_sigma.pkl', 'rb'));

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
    np.savetxt(f, Tr, fmt='%d');
    f.close();


def split_validation(classn):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(classn);

    network, encode_layer, input_var, aug_var, target_var = build_network_from_ae(classn);
    train_fn, val_fn = make_training_functions(network, encode_layer, input_var, aug_var, target_var);

    a_train = np.zeros((X_train.shape[0], classn), dtype = np.float32);
    a_val = np.zeros((X_val.shape[0], classn), dtype = np.float32);
    a_test = np.zeros((X_test.shape[0], classn), dtype = np.float32);
    for train_i in range(1):
        print("Round {}".format(train_i));
        a_train, a_val, a_test = train_round(train_fn, val_fn, classn,
            X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, y_test);
    train_round(train_fn, val_fn, classn,
            X_train, a_train, y_train, X_val, a_val, y_val, X_test, a_test, y_test);

    # Testing
    _, _, _, Er, Pr, Or, Tr = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
    save_visual_cases(X_test, Or, Tr);
    return Er, Pr, Or, Tr;


def print_auc_mean(classes, auc_mat):
    auc_mean = np.mean(auc_mat, axis = 0);
    auc_std = np.std(auc_mat, axis = 0);
    for c in range(len(classes)):
        print("{} [{:.4f}][{:.4f}]".format(classes[c], auc_mean[c], auc_std[c]));
    print("Average AUC {:.4f}".format(np.mean(auc_mean)));


def main():
    classes = ['Perinuclear halos', 'Gemistocyte', 'Nucleoli', 'Grooved', 'Clumped chromatin', 'Hyperchromasia',
               'Overlapping nuclei', 'Multinucleation', 'Severe anaplasia', 'Mitosis', 'Apoptosis', 'Other attribute',
               'No nucleus', 'Oval', 'Close to Round', 'Round', 'Elongated', 'Irregular shape', 'Other shape'];
    classn = len(classes);
    split_n = 5;
    sys.setrecursionlimit(10000);

    auc_mat = np.zeros(shape = (split_n, classn));
    for v in range(split_n):
        Er, Pr, Or, Tr = split_validation(classn);
        save_result("500x2_sp_{}".format(v), Er, Pr, Or, Tr);
        auc_vec = draw_roc("500x2_sp_{}".format(v), classes, Tr, Or);
        auc_mat[v] = auc_vec;
        print("[CNN] Hamming: {:.4f}\tAccuracy: {:.4f}".format(1 - hamming_loss(Tr, Pr), accuracy_score(Tr, Pr)));

    print_auc_mean(classes, auc_mat);


if __name__ == "__main__":
    main();

