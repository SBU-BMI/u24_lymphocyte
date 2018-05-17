import pickle
import sys
import os
import time
import lasagne
import theano
import numpy as np
import theano.tensor as T

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

APS = 100;
PS = 100;
TileFolder = sys.argv[1] + '/';
LearningRate = theano.shared(np.array(5e-3, dtype=np.float32));
BatchSize = 96;

CNNModel = sys.argv[2] + '/cnn_lym_model.pkl';
heat_map_out = sys.argv[3];

mu = 0.6151888371;
sigma = 0.2506813109;
aug_fea_n = 1;


def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def iterate_minibatches(inputs, augs, targets):
    if inputs.shape[0] <= BatchSize:
        yield inputs, augs, targets;
        return;

    start_idx = 0;
    for start_idx in range(0, len(inputs) - BatchSize + 1, BatchSize):
        excerpt = slice(start_idx, start_idx + BatchSize);
        yield inputs[excerpt], augs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - BatchSize:
        excerpt = slice(start_idx + BatchSize, len(inputs));
        yield inputs[excerpt], augs[excerpt], targets[excerpt];


def load_data(todo_list, rind):
    X = np.zeros(shape=(BatchSize*40, 3, APS, APS), dtype=np.float32);
    inds = np.zeros(shape=(BatchSize*40,), dtype=np.int32);
    coor = np.zeros(shape=(20000000, 2), dtype=np.int32);

    xind = 0;
    lind = 0;
    cind = 0;
    for fn in todo_list:
        lind += 1;
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

                if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                    X[xind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose();
                    inds[xind] = rind;
                    xind += 1;

                coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                cind += 1;
                rind += 1;

        if xind >= BatchSize:
            break;

    X = X[0:xind];
    inds = inds[0:xind];
    coor = coor[0:cind];

    return todo_list[lind:], X, inds, coor, rind;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def multi_win_during_val(val_fn, inputs, augs, targets):
    for idraw in [-1,]:
        for jdraw in [-1,]:
            ########################
            # Break batch into mini-batches
            output_pat = np.zeros((inputs.shape[0], 1), dtype=np.float32);
            ncase = 0;
            for batch in iterate_minibatches(inputs, augs, targets):
                inp, aug, tar = batch;
                _, outp = val_fn(
                        data_aug(inp, mu, sigma, deterministic=False, idraw=idraw, jdraw=jdraw),
                        aug, tar);
                output_pat[ncase:ncase+len(outp)] = outp;
                ncase += len(outp);
            # Break batch into mini-batches
            ########################

            if 'weight' in locals():
                weight += 1.0;
                output += output_pat;
            else:
                weight = 1.0;
                output = output_pat;
    return output/weight;


def val_fn_epoch_on_disk(classn, val_fn):
    all_or = np.zeros(shape=(20000000, classn), dtype=np.float32);
    all_inds = np.zeros(shape=(20000000,), dtype=np.int32);
    all_coor = np.zeros(shape=(20000000, 2), dtype=np.int32);
    rind = 0;
    n1 = 0;
    n2 = 0;
    n3 = 0;
    todo_list = os.listdir(TileFolder);
    while len(todo_list) > 0:
        todo_list, inputs, inds, coor, rind = load_data(todo_list, rind);
        if len(inputs) == 0:
            all_coor[n3:n3+len(coor)] = coor;
            n3 += len(coor);
            continue;
        augs = get_aug_feas(inputs);
        targets = np.zeros((inputs.shape[0], classn), dtype=np.int32);

        output = multi_win_during_val(val_fn, inputs, augs, targets);
        all_or[n1:n1+len(output)] = output;
        all_inds[n2:n2+len(inds)] = inds;
        all_coor[n3:n3+len(coor)] = coor;
        n1 += len(output);
        n2 += len(inds);
        n3 += len(coor);

    all_or = all_or[:n1];
    all_inds = all_inds[:n2];
    all_coor = all_coor[:n3];
    return all_or, all_inds, all_coor;


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
    mask_map = SoftThresPerc(mask_rep, perc=97.0, alpha=0.1, beta=init.Constant(0.5), tight=100.0, name="mask_map");
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

    mask_map.beta.set_value(np.float32(0.9*mask_map.beta.get_value()));
    old_params = layers.get_all_params(network, trainable=True);

    # Adding more layers
    aug_var = T.matrix('aug_var');
    target_var = T.imatrix('targets');
    add_a = batch_norm(layers.Conv2DLayer(enlyr, 320, filter_size=(1,1), nonlinearity=leaky_rectify));
    add_b = batch_norm(layers.Conv2DLayer(add_a, 320, filter_size=(1,1), nonlinearity=leaky_rectify));
    add_c = batch_norm(layers.Conv2DLayer(add_b, 320, filter_size=(1,1), nonlinearity=leaky_rectify));
    add_d = batch_norm(layers.Conv2DLayer(add_c, 320, filter_size=(1,1), nonlinearity=leaky_rectify));
    add_0 = layers.Pool2DLayer(add_d, pool_size=(25,25), stride=25, mode='average_inc_pad');
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


def split_validation(classn):
    network, new_params, input_var, aug_var, target_var = build_network_from_ae(classn);
    train_fn, new_params_train_fn, val_fn = make_training_functions(network, new_params, input_var, aug_var, target_var);
    layers.set_all_param_values(network, pickle.load(open(CNNModel, 'rb')));

    # Testing
    Or, inds, coor = val_fn_epoch_on_disk(classn, val_fn);
    Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32);
    Or_all[inds] = Or[:, 0];

    fid = open(TileFolder + '/' + heat_map_out, 'w');
    for idx in range(0, Or_all.shape[0]):
        fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx]));
    fid.close();

    return;


def main():
    if not os.path.exists(TileFolder):
        exit(0);

    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    split_validation(classn);
    print('DONE!');


if __name__ == "__main__":
    main();
