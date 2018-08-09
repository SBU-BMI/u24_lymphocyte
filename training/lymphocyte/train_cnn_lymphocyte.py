import pickle
import sys
import os
import time
import lasagne
import theano
import numpy as np
import theano.tensor as T
import gc

from lasagne import layers
from lasagne.updates import nesterov_momentum
from theano.sandbox.neighbours import neibs2images
from lasagne.nonlinearities import sigmoid, rectify, leaky_rectify, identity
from lasagne.nonlinearities import softmax
from lasagne import regularization
from scipy import misc
from PIL import Image
from lasagne import init
from math import floor
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

from data_aug import data_aug
sys.path.append('..')
from common.shape import ReshapeLayer
from common.batch_norms import batch_norm, SoftThresPerc
from common.ch_inner_prod import ChInnerProd, ChInnerProdMerge

# Training imgs should have size s: 300 <= s <= 500

APS = 300;  # ori: 300
PS = 100;
LearningRate = theano.shared(np.array(1e-3, dtype=np.float32));
BatchSize = 100;

filename_cae_model = sys.argv[1] + '/cae_model.pkl';
training_data_path = sys.argv[2];
dataset_list = training_data_path + '/lym_data_list.txt';
filename_output_model = sys.argv[3] + '/cnn_lym_model.pkl';
filename_output_model_best = sys.argv[3] + '/cnn_lym_model_best.pkl';

mu = 0.6151888371;
sigma = 0.2506813109;
aug_fea_n = 1;

print("not re-load training data frequently")

try:
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
                excerpt = indices[start_idx + batchsize: len(inputs)];
            else:
                excerpt = slice(start_idx + batchsize, len(inputs));
            yield inputs[excerpt], augs[excerpt], targets[excerpt];
except Exception as e:
    print('iterate_minibatches: ', e)
    exit(1)

try:

    def load_data_folder(classn, folder, is_train):  # only load the image filename and the labels
        img_pos = [];
        img_neg = [];
        lines = [line.rstrip('\n') for line in open(folder + '/label.txt')];
        for line in lines:
            img = line.split()[0];
            # change the label threshold to generate labels
            lab = np.array([int(int(line.split()[1]) > 0)]);
            img_file = folder + '/' + img
            if lab > 0:
                img_pos.append((img_file, lab))
            else:
                img_neg.append((img_file, lab))
        return img_pos, img_neg
except Exception as e:
    print('load_data_folder: ', e)
    exit(1)

try:
    def load_data_split(classn, folders, is_train):
        X_pos = []
        X_neg = []
        for folder in folders:
            img_pos, img_neg = load_data_folder(classn, folder, is_train);
            X_pos += img_pos
            X_neg += img_neg
        return X_pos, X_neg
except Exception as e:
    print('load_data_split: ', e)
    exit(1)

try:
    def load_imgs(img_files, classn):
        N = len(img_files)
        X = np.zeros(shape=(N, 3, APS, APS), dtype=np.float32);
        y = np.zeros(shape=(N, classn), dtype=np.int32);
        nline = 0
        for img in img_files:
            lab = np.array([int(int(img[1]) > 0)]);
            png = np.array(Image.open(img[0]).convert('RGB')).transpose();
            if (png.shape[1] >= 400):
                center = int(png.shape[1] / 2)
                png = png[:, center - APS / 2:center + APS / 2, center - APS / 2:center + APS / 2]
            if png.shape[1] == APS:
                X[nline], y[nline] = png, lab;
                nline += 1;
        X = X[0:nline];
        y = y[0:nline];
        return X, y
except Exception as e:
    print('load_imgs: ', e)
    exit(1)

try:
    def shuffle_data(data, N_limit):  # data is a list
        rands = np.random.permutation(len(data))
        out = []
        count = 0
        for i in rands:
            out.append(data[i])
            count += 1
            if count == N_limit:
                break
        return out
except Exception as e:
    print('shuffle_data: ', e)
    exit(1)

try:
    def load_data(classn):
        X_train_pos = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
        y_train_pos = np.zeros(shape=(0, classn), dtype=np.int32);
        X_train_neg = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
        y_train_neg = np.zeros(shape=(0, classn), dtype=np.int32);

        img_test_pos = []
        img_test_neg = []
        img_train_pos = []
        img_train_neg = []

        lines = [line.rstrip('\n') for line in open(dataset_list)];
        valid_i = 0;
        for line in lines:
            split_folders = [training_data_path + "/" + s for s in line.split()];
            X_pos = []
            X_neg = []
            if valid_i == 0:
                # testing data
                X_pos, X_neg = load_data_split(classn, split_folders, False);
                img_test_pos += X_pos
                img_test_neg += X_neg
            else:
                # training dataX_pos
                X_pos, X_neg = load_data_split(classn, split_folders, True);
                img_train_pos += X_pos
                img_train_neg += X_neg

            valid_i += 1;

        # ========== shuffle train_data, no need to shuffle test data ========
        N_pos = len(img_train_pos)
        N_neg = len(img_train_neg)

        #    if N_neg > N_pos:
        #        img_train_neg = shuffle_data(img_train_neg, min(N_neg, 2*N_pos))

        img_trains = img_train_pos + img_train_neg
        img_trains = shuffle_data(img_trains, len(img_trains))
        X_train, y_train = load_imgs(img_trains, classn)

        img_trains = None;
        img_train_neg = None;
        img_train_pos = None;
        X_pos = None;
        X_neg = None;
        gc.collect()
        # ==== testing data ====
        img_vals = img_test_pos + img_test_neg
        img_test_pos = None;
        img_test_neg = None;
        gc.collect();
        X_test, y_test = load_imgs(img_vals, classn)

        print "Data Loaded", X_train.shape, y_train.shape, X_test.shape, y_test.shape;
        return X_train, y_train, X_test, y_test;

except Exception as e:
    print('load_data: ', e)
    exit(1)

try:
    def from_output_to_pred(output):
        pred = np.copy(output);
        pred = (pred >= 0.5).astype(np.int32);
        return pred;

except Exception as e:
    print('from_output_to_pred: ', e)
    exit(1)

try:
    def multi_win_during_val(val_fn, inputs, augs, targets):
        for idraw in [50, 75, 100, 125, 150]:  # ori: [50, 75, 100, 125, 150]
            for jdraw in [50, 75, 100, 125, 150]:  # ori: [50, 75, 100, 125, 150]
                inpt_multiwin = data_aug(inputs, mu, sigma, deterministic=True, idraw=idraw, jdraw=jdraw);
                err_pat, output_pat = val_fn(inpt_multiwin, augs, targets);
                if 'weight' in locals():
                    dis = ((idraw / 100.0 - 1.0) ** 2 + (jdraw / 100.0 - 1.0) ** 2) ** 0.5;
                    wei = np.exp(-np.square(dis) / 2.0 / 0.5 ** 2);
                    weight += wei;
                    err += err_pat * wei;
                    output += output_pat * wei;
                else:
                    dis = ((idraw / 100.0 - 1.0) ** 2 + (jdraw / 100.0 - 1.0) ** 2) ** 0.5;
                    weight = np.exp(-np.square(dis) / 2.0 / 1.0 ** 2);
                    err = err_pat * weight;
                    output = output_pat * weight;
        return err / weight, output / weight;
except Exception as e:
    print('multi_win_during_val: ', e)
    exit(1)

try:
    def val_fn_epoch(classn, val_fn, X_val, a_val, y_val):
        val_err = 0;
        Pr = np.empty(shape=(10000, classn), dtype=np.int32);
        Or = np.empty(shape=(10000, classn), dtype=np.float32);
        Tr = np.empty(shape=(10000, classn), dtype=np.int32);
        val_batches = 0;
        nline = 0;
        for batch in iterate_minibatches(X_val, a_val, y_val, BatchSize, shuffle=False):
            inputs, augs, targets = batch;
            err, output = multi_win_during_val(val_fn, inputs, augs, targets);
            pred = from_output_to_pred(output);
            val_err += err;
            Pr[nline:nline + len(output)] = pred;
            Or[nline:nline + len(output)] = output;
            Tr[nline:nline + len(output)] = targets;
            val_batches += 1;
            nline += len(output);
        Pr = Pr[:nline];
        Or = Or[:nline];
        Tr = Tr[:nline];
        val_err = val_err / val_batches;
        val_ham = (1 - hamming_loss(Tr, Pr));
        val_acc = accuracy_score(Tr, Pr);
        return val_err, val_ham, val_acc, Pr, Or, Tr;
except Exception as e:
    print('val_fn_epoch: ', e)
    exit(1)

try:
    def confusion_matrix(Or, Tr, thres):
        tpos = np.sum((Or >= thres) * (Tr == 1));
        tneg = np.sum((Or < thres) * (Tr == 0));
        fpos = np.sum((Or >= thres) * (Tr == 0));
        fneg = np.sum((Or < thres) * (Tr == 1));
        return tpos, tneg, fpos, fneg;

except Exception as e:
    print('confusion_matrix: ', e)
    exit(1)

try:
    def auc_roc(Pr, Tr):
        fpr, tpr, _ = roc_curve(Tr, Pr, pos_label=1.0);
        return auc(fpr, tpr);
except Exception as e:
    print('auc_roc: ', e)
    exit(1)

try:
    def train_round(num_epochs, network, train_fn, val_fn, classn, X_train, a_train, y_train, X_test, a_test,
                    y_test):  #
        val_auc_best = 0
        print("Starting training...")
        print("tpos, tneg, fpos, fneg")
        print("TrLoss\tVaLoss\tAUC\tCMatrix0\tCMatrix1\tCMatrix2\tEpochs\tTime");
        start_time = time.time();
        for epoch in range(num_epochs + 1):

            train_err = 0;
            train_batches = 0;
            for batch in iterate_minibatches(X_train, a_train, y_train, BatchSize, shuffle=True):
                inputs, augs, targets = batch;
                inputs = data_aug(inputs, mu, sigma);
                train_err += train_fn(inputs, augs, targets);
                train_batches += 1;
            train_err = train_err / train_batches;

            if epoch % 1 == 0:
                # And a full pass over the validation data:
                test_err, _, _, _, Or, Tr = val_fn_epoch(classn, val_fn, X_test, a_test, y_test);
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
                    epoch + 1, num_epochs, time.time() - start_time));
                start_time = time.time();
            if val_auc_best < val_auc:
                val_auc_best = val_auc
                param_values = layers.get_all_param_values(network);
                pickle.dump(param_values, open(filename_output_model_best, 'w'));
            if (epoch + 1) % 2 == 0:
                param_values = layers.get_all_param_values(network);
                pickle.dump(param_values, open(filename_output_model, 'w'));

            if epoch == 5:
                LearningRate.set_value(np.float32(0.50 * LearningRate.get_value()));

        param_values = layers.get_all_param_values(network);
        pickle.dump(param_values, open(filename_output_model, 'w'));
except Exception as e:
    print('train_round: ', e)
    exit(1)

try:
    def build_network_from_ae(classn):
        input_var = T.tensor4('input_var');

        layer = layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);
        layer = batch_norm(
            layers.Conv2DLayer(layer, 100, filter_size=(5, 5), stride=1, pad='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Conv2DLayer(layer, 120, filter_size=(5, 5), stride=1, pad='same', nonlinearity=leaky_rectify));
        layer = layers.Pool2DLayer(layer, pool_size=(2, 2), stride=2, mode='average_inc_pad');
        layer = batch_norm(
            layers.Conv2DLayer(layer, 240, filter_size=(3, 3), stride=1, pad='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Conv2DLayer(layer, 320, filter_size=(3, 3), stride=1, pad='same', nonlinearity=leaky_rectify));
        layer = layers.Pool2DLayer(layer, pool_size=(2, 2), stride=2, mode='average_inc_pad');
        layer = batch_norm(
            layers.Conv2DLayer(layer, 640, filter_size=(3, 3), stride=1, pad='same', nonlinearity=leaky_rectify));
        prely = batch_norm(
            layers.Conv2DLayer(layer, 1024, filter_size=(3, 3), stride=1, pad='same', nonlinearity=leaky_rectify));

        featm = batch_norm(layers.Conv2DLayer(prely, 640, filter_size=(1, 1), nonlinearity=leaky_rectify));
        feat_map = batch_norm(
            layers.Conv2DLayer(featm, 100, filter_size=(1, 1), nonlinearity=rectify, name="feat_map"));
        maskm = batch_norm(layers.Conv2DLayer(prely, 100, filter_size=(1, 1), nonlinearity=leaky_rectify));
        mask_rep = batch_norm(layers.Conv2DLayer(maskm, 1, filter_size=(1, 1), nonlinearity=None), beta=None,
                              gamma=None);
        mask_map = SoftThresPerc(mask_rep, perc=97.0, alpha=0.1, beta=init.Constant(0.5), tight=100.0, name="mask_map");
        enlyr = ChInnerProdMerge(feat_map, mask_map, name="encoder");

        layer = batch_norm(
            layers.Deconv2DLayer(enlyr, 1024, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 640, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 640, filter_size=(4, 4), stride=2, crop=(1, 1), nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 320, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 320, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 240, filter_size=(4, 4), stride=2, crop=(1, 1), nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 120, filter_size=(5, 5), stride=1, crop='same', nonlinearity=leaky_rectify));
        layer = batch_norm(
            layers.Deconv2DLayer(layer, 100, filter_size=(5, 5), stride=1, crop='same', nonlinearity=leaky_rectify));
        layer = layers.Deconv2DLayer(layer, 3, filter_size=(1, 1), stride=1, crop='same', nonlinearity=identity);

        glblf = batch_norm(layers.Conv2DLayer(prely, 128, filter_size=(1, 1), nonlinearity=leaky_rectify));
        glblf = layers.Pool2DLayer(glblf, pool_size=(5, 5), stride=5, mode='average_inc_pad');
        glblf = batch_norm(
            layers.Conv2DLayer(glblf, 64, filter_size=(3, 3), stride=1, pad='same', nonlinearity=leaky_rectify));
        gllyr = batch_norm(layers.Conv2DLayer(glblf, 5, filter_size=(1, 1), nonlinearity=rectify),
                           name="global_feature");

        glblf = batch_norm(
            layers.Deconv2DLayer(gllyr, 256, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 128, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 128, filter_size=(9, 9), stride=5, crop=(2, 2), nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 128, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 128, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 64, filter_size=(4, 4), stride=2, crop=(1, 1), nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 64, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 64, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 32, filter_size=(4, 4), stride=2, crop=(1, 1), nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 32, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = batch_norm(
            layers.Deconv2DLayer(glblf, 32, filter_size=(3, 3), stride=1, crop='same', nonlinearity=leaky_rectify));
        glblf = layers.Deconv2DLayer(glblf, 3, filter_size=(1, 1), stride=1, crop='same', nonlinearity=identity);

        layer = layers.ElemwiseSumLayer([layer, glblf]);

        network = ReshapeLayer(layer, ([0], -1));
        layers.set_all_param_values(network, pickle.load(open(filename_cae_model, 'rb')));
        mask_map.beta.set_value(np.float32(0.9 * mask_map.beta.get_value()));
        old_params = layers.get_all_params(network, trainable=True);

        # Adding more layers
        aug_var = T.matrix('aug_var');
        target_var = T.imatrix('targets');
        add_a = batch_norm(layers.Conv2DLayer(enlyr, 320, filter_size=(1, 1), nonlinearity=leaky_rectify));
        add_b = batch_norm(layers.Conv2DLayer(add_a, 320, filter_size=(1, 1), nonlinearity=leaky_rectify));
        add_c = batch_norm(layers.Conv2DLayer(add_b, 320, filter_size=(1, 1), nonlinearity=leaky_rectify));
        add_d = batch_norm(layers.Conv2DLayer(add_c, 320, filter_size=(1, 1), nonlinearity=leaky_rectify));
        add_0 = layers.Pool2DLayer(add_d, pool_size=(25, 25), stride=25, mode='average_inc_pad');
        add_1 = batch_norm(layers.DenseLayer(add_0, 100, nonlinearity=leaky_rectify));

        add_2 = batch_norm(layers.DenseLayer(gllyr, 320, nonlinearity=leaky_rectify));
        add_3 = batch_norm(layers.DenseLayer(add_2, 320, nonlinearity=leaky_rectify));
        add_4 = batch_norm(layers.DenseLayer(add_3, 100, nonlinearity=leaky_rectify));

        aug_layer = layers.InputLayer(shape=(None, aug_fea_n), input_var=aug_var);

        cat_layer = lasagne.layers.ConcatLayer([add_1, add_4, aug_layer], axis=1);

        hidden_layer = layers.DenseLayer(cat_layer, 80, nonlinearity=leaky_rectify);
        network = layers.DenseLayer(hidden_layer, classn, nonlinearity=sigmoid);

        all_params = layers.get_all_params(network, trainable=True);
        new_params = [x for x in all_params if x not in old_params];

        return network, new_params, input_var, aug_var, target_var;
except Exception as e:
    print('build_network_from_ae: ', e)
    exit(1)

try:
    def make_training_functions(network, new_params, input_var, aug_var, target_var):
        output = lasagne.layers.get_output(network);
        loss = lasagne.objectives.binary_crossentropy(output, target_var).mean();

        deter_output = lasagne.layers.get_output(network, deterministic=True);
        deter_loss = lasagne.objectives.binary_crossentropy(deter_output, target_var).mean();

        params = layers.get_all_params(network, trainable=True);
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=LearningRate, momentum=0.985);
        new_params_updates = lasagne.updates.nesterov_momentum(loss, new_params, learning_rate=LearningRate,
                                                               momentum=0.985);

        val_fn = theano.function([input_var, aug_var, target_var], [deter_loss, deter_output]);
        train_fn = theano.function([input_var, aug_var, target_var], loss, updates=updates);
        new_params_train_fn = theano.function([input_var, aug_var, target_var], loss, updates=new_params_updates);

        return train_fn, new_params_train_fn, val_fn;
except Exception as e:
    print('make_training_functions: ', e)
    exit(1)

try:
    def get_aug_feas(X):
        aug_feas = np.zeros((X.shape[0], aug_fea_n), dtype=np.float32);
        return aug_feas;

except Exception as e:
    print('get_aug_feas: ', e)
    exit(1)

try:
    def split_validation(classn):
        network, new_params, input_var, aug_var, target_var = build_network_from_ae(classn);
        train_fn, new_params_train_fn, val_fn = make_training_functions(network, new_params, input_var, aug_var,
                                                                        target_var);

        X_train, y_train, X_test, y_test = load_data(classn);
        a_train = get_aug_feas(X_train);
        a_test = get_aug_feas(X_test);
        train_round(20, network, new_params_train_fn, val_fn, classn, X_train, a_train, y_train, X_test, a_test,
                    y_test);  #
        LearningRate.set_value(np.float32(0.10 * LearningRate.get_value()));
        train_round(20, network, train_fn, val_fn, classn, X_train, a_train, y_train, X_test, a_test,
                    y_test);  # , X_train, a_train, y_train, X_test, a_test, y_test

        return;
except Exception as e:
    print('split_validation: ', e)
    exit(1)

def main():
    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    split_validation(classn);


if __name__ == "__main__":
    main();
