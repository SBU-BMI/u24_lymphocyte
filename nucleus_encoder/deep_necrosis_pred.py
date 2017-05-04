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

APS = 500;
PS = 224;
TrainFolder = '/data08/shared/lehhou/data/necrosis/train_Apr10_dirtynecrosis_train/';
#TileFolder = sys.argv[1] + '/';
TileFolder = '/data08/shared/lehhou/nucleus_encoder/test_data/TCGA-VS-A9UR-01Z-00-DX1.8475AB5B-6699-4F92-B5F8-7B8A7DF61942.svs/';
list_file = TileFolder + 'list.txt' #???
LearningRate = theano.shared(np.array(5e-3, dtype=np.float32));
BatchSize = 10;

#filename_code = 21;
filesave_code = 410;
filename_model_ae = 'model_vals/deep_conv_classification_model_deep_segmentation_deconv_necrosis_alt2_Apr11_e{}.pkl'.format(filesave_code);
#filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_mu.pkl'.format(filename_code);
#filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt{}_sigma.pkl'.format(filename_code);

mu = None;
sigma = None;
aug_fea_n = 1;

read_epoch = 10;
read_cv = 0;
heat_map_out = 'patch-level-pred-3-21-paad.txt';
CNNModel = 'model_vals/' + sys.argv[0].split('_heatmap.py')[0] + '.py_e{}_cv{}.pkl'.format(read_epoch, read_cv);


def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


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

def read_image_list_file(text_file_path):
    with open(text_file_path) as f:
        content = f.readlines();

    content = [x.strip() for x in content];
    return content;

def load_train_data():
    X = np.zeros(shape=(10000, 3, APS, APS), dtype=np.float32);
    Y_raw = np.zeros(shape=(10000, 3, APS, APS), dtype=np.float32);
    Y = np.zeros(shape=(10000, 1, APS, APS), dtype=np.float32);
    inds = np.zeros(shape=(10000,), dtype=np.int32);

    xind = 0;
    cind = 0;
    image_names = read_image_list_file(TrainFolder + 'list.txt');
    for fn in image_names:
        full_fn = TrainFolder + '/' + fn;
        mask_fn = TrainFolder + '/mask_' + fn[6:];
        png = np.array(Image.open(full_fn).convert('RGB'));
        png_Y = np.array(Image.open(mask_fn).convert('RGB'));
        msk = np.array(Image.open(full_fn[
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue;
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue;

                #if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                X[xind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose();
                Y_raw[xind, :, :, :] = png_Y[y:y+APS, x:x+APS, :].transpose();
                inds[xind] = cind;
                xind += 1;

                #coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                #coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                cind += 1;

        break;

    X = X[0:xind];
    Y = Y_raw[0:xind,1,:,:];
    inds = inds[0:xind];
    #coor = coor[0:cind];

    print "Data Loaded", X.shape, inds.shape;
    return X, Y, inds;


def load_data():
    X = np.zeros(shape=(800000, 3, APS, APS), dtype=np.float32);
    inds = np.zeros(shape=(800000,), dtype=np.int32);
    coor = np.zeros(shape=(1000000, 2), dtype=np.int32);

    xind = 0;
    cind = 0;
    image_names = read_image_list_file(list_file);
    #for fn in os.listdir(TileFolder):
    for fn in image_names:
        full_fn = TileFolder + '/' + fn;
        if not os.path.isfile(full_fn):
            continue;
        #if len(fn.split('_')) < 4:
            #continue;

        #x_off = float(fn.split('_')[0]);
        #y_off = float(fn.split('_')[1]);
        #svs_pw = float(fn.split('_')[2]);
        #png_pw = float(fn.split('_')[3].split('.png')[0]);

        png = np.array(Image.open(full_fn).convert('RGB'));
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue;
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue;

                #if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                X[xind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose();
                inds[xind] = cind;
                xind += 1;

                #coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                #coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                cind += 1;

        break;

    X = X[0:xind];
    inds = inds[0:xind];
    #coor = coor[0:cind];

    print "Data Loaded", X.shape, inds.shape;
    return X, inds;


def from_output_to_pred(output):
    pred = np.copy(output);
    pred = (pred >= 0.5).astype(np.int32);
    return pred;


def multi_win_during_val(val_fn, inputs, augs, targets):
    for idraw in [-1,]:
        for jdraw in [-1,]:
            inpt_multiwin = data_aug(inputs, mu, sigma, deterministic=True, idraw=idraw, jdraw=jdraw);
            err_pat, output_pat = val_fn(inpt_multiwin, augs, targets);
            """
            if 'weight' in locals():
                weight += 1.0;
                err += err_pat;
                output += output_pat;
            else:
                weight = 1.0;
                err = err_pat;
                output = output_pat;
            """
    #return err/weight, output/weight;
    return output;


def val_fn_epoch(classn, val_fn, X_val, y_val):
    val_err = 0;
    Pr = np.empty(shape = (50000, PS*PS), dtype = np.int32);
    Or = np.empty(shape = (50000, PS*PS), dtype = np.float32);
    Tr = np.empty(shape = (50000, PS*PS), dtype = np.int32);
    val_batches = 0;
    nline = 0;
    for batch in iterate_minibatches(X_val, y_val, BatchSize, shuffle = False):
        inputs, targets = batch;
        output = multi_win_during_val(val_fn, inputs, augs, targets);
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
    #val_err = val_err / val_batches;
    #val_ham = (1 - hamming_loss(Tr, Pr));
    #val_acc = accuracy_score(Tr, Pr);
    return Pr, Or, Tr;


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

    net = {}

    net['input'] =  layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);

    net['conv1_1'] = batch_norm(layers.Conv2DLayer(net['input'], 64,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv1_2'] = batch_norm(layers.Conv2DLayer(net['conv1_1'], 64,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['pool1'] = layers.Pool2DLayer(net['conv1_2'], pool_size=(2,2), stride=2, mode='max');

    net['conv2_1'] = batch_norm(layers.Conv2DLayer(net['pool1'], 128,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv2_2'] = batch_norm(layers.Conv2DLayer(net['conv2_1'], 128,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['pool2'] = layers.Pool2DLayer(net['conv2_2'], pool_size=(2,2), stride=2, mode='max');

    net['conv3_1'] = batch_norm(layers.Conv2DLayer(net['pool2'], 256,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv3_2'] = batch_norm(layers.Conv2DLayer(net['conv3_1'], 256,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv3_3'] = batch_norm(layers.Conv2DLayer(net['conv3_2'], 256,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['pool3'] = layers.Pool2DLayer(net['conv3_3'], pool_size=(2,2), stride=2, mode='max');

    net['conv4_1'] = batch_norm(layers.Conv2DLayer(net['pool3'], 512,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv4_2'] = batch_norm(layers.Conv2DLayer(net['conv4_1'], 512,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv4_3'] = batch_norm(layers.Conv2DLayer(net['conv4_2'], 512,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['pool4'] = layers.Pool2DLayer(net['conv4_3'], pool_size=(2,2), stride=2, mode='max');

    net['conv5_1'] = batch_norm(layers.Conv2DLayer(net['pool4'], 512,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv5_2'] = batch_norm(layers.Conv2DLayer(net['conv5_1'], 512,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['conv5_3'] = batch_norm(layers.Conv2DLayer(net['conv5_2'], 512,  filter_size=(3,3), stride=1, pad='same', nonlinearity=leaky_rectify));
    net['pool5'] = layers.Pool2DLayer(net['conv5_3'], pool_size=(2,2), stride=2, mode='max');

    net['fc6'] = batch_norm(layers.Conv2DLayer(net['pool5'], 4096,  filter_size=(7,7), stride=1, pad='valid', nonlinearity=leaky_rectify));
    net['fc7'] = batch_norm(layers.Conv2DLayer(net['fc6'], 4096,  filter_size=(1,1), stride=1, pad='valid', nonlinearity=leaky_rectify));

    net['fc6_deconv'] = batch_norm(layers.Deconv2DLayer(net['fc7'], 512, filter_size=(7,7), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['unpool5'] = layers.InverseLayer(net['fc6_deconv'], net['pool5']);

    net['deconv5_1'] = batch_norm(layers.Deconv2DLayer(net['unpool5'], 512, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv5_2'] = batch_norm(layers.Deconv2DLayer(net['deconv5_1'], 512, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv5_3'] = batch_norm(layers.Deconv2DLayer(net['deconv5_2'], 512, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['unpool4'] = layers.InverseLayer(net['deconv5_3'], net['pool4']);

    net['deconv4_1'] = batch_norm(layers.Deconv2DLayer(net['unpool4'], 512, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv4_2'] = batch_norm(layers.Deconv2DLayer(net['deconv4_1'], 512, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv4_3'] = batch_norm(layers.Deconv2DLayer(net['deconv4_2'], 256, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['unpool3'] = layers.InverseLayer(net['deconv4_3'], net['pool3']);

    net['deconv3_1'] = batch_norm(layers.Deconv2DLayer(net['unpool3'], 256, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv3_2'] = batch_norm(layers.Deconv2DLayer(net['deconv3_1'], 256, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv3_3'] = batch_norm(layers.Deconv2DLayer(net['deconv3_2'], 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['unpool2'] = layers.InverseLayer(net['deconv3_3'], net['pool2']);

    net['deconv2_1'] = batch_norm(layers.Deconv2DLayer(net['unpool2'], 128, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv2_2'] = batch_norm(layers.Deconv2DLayer(net['deconv2_1'], 64, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['unpool1'] = layers.InverseLayer(net['deconv2_2'], net['pool1']);

    net['deconv1_1'] = batch_norm(layers.Deconv2DLayer(net['unpool1'], 64, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));
    net['deconv1_2'] = batch_norm(layers.Deconv2DLayer(net['deconv1_1'], 64, filter_size=(3,3), stride=1, crop='same', nonlinearity=leaky_rectify));

    net['score'] = layers.Deconv2DLayer(net['deconv1_2'], 1,  filter_size=(1,1), stride=1, crop='same', nonlinearity=None);
    net['score_flat'] = ReshapeLayer(net['score'], ([0], -1));

    all_params = layers.get_all_params(net['score_flat'], trainable=True);
    target_var = T.fmatrix('targets');

    return net, all_params, input_var, target_var;


def build_network_from_ae_old(classn):
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
    mask_map.beta.set_value(np.float32(0.8*mask_map.beta.get_value()));
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

def make_training_functions(network, all_params, input_var, target_var):
    # ??? what are batch_norm_use_averages and batch_norm_update_averages
    #output = lasagne.layers.get_output(network, deterministic=True, batch_norm_use_averages=True, batch_norm_update_averages=False);
    output = lasagne.layers.get_output(network['score_flat'], deterministic=False);
    loss = lasagne.objectives.binary_crossentropy(output, target_var).mean();

    deter_output = lasagne.layers.get_output(network['score_flat'], deterministic=True);
    deter_loss = lasagne.objectives.binary_crossentropy(deter_output, target_var).mean();

    #params = layers.get_all_params(network, trainable=True);
    updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=LearningRate, momentum=0.985);
    #new_params_updates = lasagne.updates.nesterov_momentum(loss, new_params, learning_rate=LearningRate, momentum=0.985);

    val_fn = theano.function([input_var, target_var], [deter_loss, deter_output]);
    train_fn = theano.function([input_var, target_var], loss, updates=updates);
    #new_params_train_fn = theano.function([input_var, aug_var, target_var], loss, updates=new_params_updates);

    return train_fn, val_fn;


def get_aug_feas(X):
    aug_feas = np.zeros((X.shape[0], aug_fea_n), dtype=np.float32);
    return aug_feas;

def write_to_image(pred):
    # abc
    for idx in range(pred.shape[0]):
        filename = 'necrosis_img/' +
        scipy.misc.imsave('outfile.jpg', image_array);


def split_validation(classn):
    #X, inds, coor = load_data();
    X, inds = load_train_data();

    network, all_params, input_var, target_var = build_network_from_ae(classn);
    train_fn, val_fn = make_training_functions(network, all_params, input_var, target_var);

    loaded_var = pickle.load(open(filename_model_ae, 'rb'));
    mu = loaded_var[0];
    sigma = loaded_var[1];
    loaded_params = loaded_var[2];


    layers.set_all_param_values(network['score_flat'], loaded_params);

    Pr, Or, Tr = val_fn_epoch(classn, val_fn, X, Y);

    """
    A = get_aug_feas(X);
    Y = np.zeros((X.shape[0], classn), dtype=np.int32);

    # Testing
    _, _, _, _, Or, _ = val_fn_epoch(classn, val_fn, X, A, Y);
    Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32);
    Or_all[inds] = Or[:, 0];

    fid = open(TileFolder + '/' + heat_map_out, 'w');
    for idx in range(0, Or_all.shape[0]):
        fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx]));
    fid.close();

    """
    return;


def main():
    if not os.path.exists(TileFolder):
        exit(0);

    #check_ext_file = TileFolder + '/' + heat_map_out;
    #if os.path.isfile(check_ext_file):
    #    exit(0);

    classes = ['Necrosis'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    load_train_data()
    #split_validation(classn);
    print('DONE!');


if __name__ == "__main__":
    main();

