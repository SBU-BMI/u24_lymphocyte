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
from extensive_data_aug_500x500 import data_aug
from ch_inner_prod import ChInnerProd, ChInnerProdMerge

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc

import scipy.misc

APS = 500;
PS = 200;
TrainFolder = '/data08/shared/lehhou/data/necrosis/train_Apr10_dirtynecrosis_train/';
#TileFolder = sys.argv[1] + '/';
TileFolder = '/data08/shared/lehhou/nucleus_encoder/test_data/train_Apr10_old_train/';
list_file = TileFolder + 'list.txt' #???
LearningRate = theano.shared(np.array(5e-3, dtype=np.float32));
BatchSize = 1;

#filename_code = 21;
filesave_code = 270;
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
    image_names = read_image_list_file(TileFolder + 'list.txt');
    print "Number of lines in the list: ", len(image_names);
    for fn in image_names:
        full_fn = TileFolder + '/' + fn;
        mask_fn = TileFolder + '/mask_' + fn[6:];
        print mask_fn;
        png = np.array(Image.open(full_fn).convert('RGB'));
        png_Y = np.array(Image.open(mask_fn).convert('RGB'));
        if (np.sum(png_Y) == 0):
            print "2222222222222222222";
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

        #break;

    X = X[0:xind];
    Y = Y_raw[0:xind,1,:,:];
    inds = inds[0:xind];
    #coor = coor[0:cind];

    if (np.sum(Y) == 0):
        print "1111111111111111111111111";

    print "Data Loaded", X.shape, inds.shape;
    return X, Y, inds;


def load_data():
    X = np.zeros(shape=(800000, 3, APS, APS), dtype=np.float32);
    inds = np.zeros(shape=(800000,), dtype=np.int32);
    coor = np.zeros(shape=(800000, 2), dtype=np.int32);

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


def multi_win_during_val(val_fn, inputs, targets):
    for idraw in [-1,]:
        for jdraw in [-1,]:
            inpt_multiwin, output = data_aug(inputs, targets, mu, sigma, deterministic=True, idraw=idraw, jdraw=jdraw, APS=APS, PS=PS);
            output_gt = np.reshape(output, (output.shape[0], -1));
            print "input, output ", inpt_multiwin.shape, output_gt.shape;
            #err_pat, output_pat = val_fn(inpt_multiwin, output);
            output_pat = val_fn(inpt_multiwin);
            print output_pat;
            print "generated dimension ", output_pat[0].shape;
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
    return inpt_multiwin, output_gt, output_pat;


def val_fn_epoch(classn, val_fn, X_val, y_val):
    val_err = 0;
    Ip = np.empty(shape = (50000, 3, PS, PS), dtype = np.float32);
    Pr = np.empty(shape = (50000, PS*PS), dtype = np.int32);
    Or = np.empty(shape = (50000, PS*PS), dtype = np.float32);
    Tr = np.empty(shape = (50000, PS*PS), dtype = np.float32);
    val_batches = 0;
    nline = 0;
    for batch in iterate_minibatches(X_val, y_val, BatchSize, shuffle = False):
        inputs, targets = batch;
        if (np.sum(targets) == 0):
            print "00000000000000000000000000";

        img, groundtruth, output = multi_win_during_val(val_fn, inputs, targets);
        if (np.sum(groundtruth) == 0):
            print "00000000000000000000000000";

        pred = from_output_to_pred(output);
        #val_err += err;
        Ip[nline:nline+len(output)] = img;
        Pr[nline:nline+len(output)] = pred;
        Or[nline:nline+len(output)] = output;
        Tr[nline:nline+len(output)] = groundtruth;
        val_batches += 1;
        nline += len(output);
    Pr = Pr[:nline];
    Or = Or[:nline];
    Tr = Tr[:nline];
    #val_err = val_err / val_batches;
    #val_ham = (1 - hamming_loss(Tr, Pr));
    #val_acc = accuracy_score(Tr, Pr);
    return Ip, Pr, Or, Tr;


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
    input_var = theano.tensor.tensor4('input_var');

    net = {};
    net['input'] = layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);


    # Encoding part
    net['conv1_1'] = batch_norm(layers.Conv2DLayer(net['input'], 64,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv1_2'] = batch_norm(layers.Conv2DLayer(net['conv1_1'], 64,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['pool1'] = layers.Pool2DLayer(net['conv1_2'], pool_size=(2,2), stride=2, mode='max');

    net['conv2_1'] = batch_norm(layers.Conv2DLayer(net['pool1'], 128,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv2_2'] = batch_norm(layers.Conv2DLayer(net['conv2_1'], 128,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['pool2'] = layers.Pool2DLayer(net['conv2_2'], pool_size=(2,2), stride=2, mode='max');

    net['conv3_1'] = batch_norm(layers.Conv2DLayer(net['pool2'], 256,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv3_2'] = batch_norm(layers.Conv2DLayer(net['conv3_1'], 256,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv3_3'] = batch_norm(layers.Conv2DLayer(net['conv3_2'], 256,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['pool3'] = layers.Pool2DLayer(net['conv3_3'], pool_size=(2,2), stride=2, mode='max');

    net['conv4_1'] = batch_norm(layers.Conv2DLayer(net['pool3'], 512,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv4_2'] = batch_norm(layers.Conv2DLayer(net['conv4_1'], 512,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv4_3'] = batch_norm(layers.Conv2DLayer(net['conv4_2'], 512,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['pool4'] = layers.Pool2DLayer(net['conv4_3'], pool_size=(2,2), stride=2, mode='max');

    net['conv5_1'] = batch_norm(layers.Conv2DLayer(net['pool4'], 512,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv5_2'] = batch_norm(layers.Conv2DLayer(net['conv5_1'], 512,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['conv5_3'] = batch_norm(layers.Conv2DLayer(net['conv5_2'], 512,   filter_size=(3,3), stride=1, pad=1, nonlinearity=leaky_rectify));
    net['pool5'] = layers.Pool2DLayer(net['conv5_3'], pool_size=(2,2), stride=2, mode='max');

    net['fc6'] = batch_norm(layers.Conv2DLayer(net['pool5'], 4096,   filter_size=(7,7), stride=1, pad='same', nonlinearity=leaky_rectify));

    # fc7 is the encoding layer
    net['fc7'] = batch_norm(layers.Conv2DLayer(net['fc6'], 4096,   filter_size=(1,1), stride=1, pad='same', nonlinearity=leaky_rectify));

    # Decoding part
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


    # Segmentation layer
    net['seg_score'] = layers.Deconv2DLayer(net['deconv1_2'], 1, filter_size=(1,1), stride=1, crop='same', nonlinearity=lasagne.nonlinearities.sigmoid);

    net['score_flat'] = ReshapeLayer(net['seg_score'], ([0], -1));
    output_var = lasagne.layers.get_output(net['score_flat']);
    all_param = lasagne.layers.get_all_params(net['score_flat'], trainable=True);

    target_var = T.fmatrix('targets');
    #return network, input_var, output_var, all_param;
    return net, all_param, input_var, target_var;

def build_network_from_ae_old(classn):
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
    #net['fc6'] = batch_norm(layers.DenseLayer(net['pool5'], 4096, nonlinearity=leaky_rectify));
    #net['fc7'] = batch_norm(layers.DenseLayer(net['fc6'], 4096, nonlinearity=leaky_rectify));

    net['fc6_deconv'] = batch_norm(layers.Deconv2DLayer(net['fc7'], 512, filter_size=(7,7), stride=1, crop='valid', nonlinearity=leaky_rectify));
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

    net['score'] = layers.Deconv2DLayer(net['deconv1_2'], 1,  filter_size=(1,1), stride=1, crop='same', nonlinearity=sigmoid);
    net['score_flat'] = ReshapeLayer(net['score'], ([0], -1));

    all_params = layers.get_all_params(net['score_flat'], trainable=True);
    target_var = T.fmatrix('targets');

    return net, all_params, input_var, target_var;



def make_training_functions(network, all_params, input_var, target_var):
    # ??? what are batch_norm_use_averages and batch_norm_update_averages
    #output = lasagne.layers.get_output(network, deterministic=True, batch_norm_use_averages=True, batch_norm_update_averages=False);
    output = lasagne.layers.get_output(network['score_flat'], deterministic=False);
    loss = lasagne.objectives.binary_crossentropy(output, target_var).mean();

    deter_output = lasagne.layers.get_output(network['score_flat'], deterministic=True);
    deter_output_middle = lasagne.layers.get_output(network['score_flat'], deterministic=True);
    deter_loss = lasagne.objectives.binary_crossentropy(deter_output, target_var).mean();

    #params = layers.get_all_params(network, trainable=True);
    updates = lasagne.updates.nesterov_momentum(loss, all_params, learning_rate=LearningRate, momentum=0.985);
    #new_params_updates = lasagne.updates.nesterov_momentum(loss, new_params, learning_rate=LearningRate, momentum=0.985);

    #val_fn = theano.function([input_var, target_var], [deter_loss, deter_output]);
    val_fn = theano.function([input_var], [deter_output_middle]);
    train_fn = theano.function([input_var, target_var], loss, updates=updates);
    #new_params_train_fn = theano.function([input_var, aug_var, target_var], loss, updates=new_params_updates);

    return train_fn, val_fn;


def get_aug_feas(X):
    aug_feas = np.zeros((X.shape[0], aug_fea_n), dtype=np.float32);
    return aug_feas;

def write_to_image(img, gt, pred):
    # abc
    print "write to image ", pred.shape;
    for idx in range(pred.shape[0]):
        written = img[idx].transpose();
        filename = './necrosis_test_img/image_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);

        written = np.reshape(gt[idx], (PS, PS)).transpose();
        filename = './necrosis_test_img/gt_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);

        written = np.reshape(pred[idx], (PS, PS)).transpose();
        filename = './necrosis_test_img/pred_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);


def split_validation(classn):
    #X, inds, coor = load_data();
    X, Y, inds = load_train_data();

    network, all_params, input_var, target_var = build_network_from_ae(classn);
    train_fn, val_fn = make_training_functions(network, all_params, input_var, target_var);

    global mu;
    global sigma;
    loaded_var = pickle.load(open(filename_model_ae, 'rb'));
    mu = loaded_var[0];
    sigma = loaded_var[1];
    loaded_params = loaded_var[2];


    layers.set_all_param_values(network['score_flat'], loaded_params);

    Ip, Pr, Or, Tr = val_fn_epoch(classn, val_fn, X, Y);
    write_to_image(Ip, Tr, Or);

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

    #load_train_data()
    split_validation(classn);
    print('DONE!');


if __name__ == "__main__":
    main();

