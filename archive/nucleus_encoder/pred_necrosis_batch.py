import pickle
import numpy as np
import lasagne
from lasagne import layers
from lasagne.nonlinearities import sigmoid, rectify, leaky_rectify, identity
import time
from extensive_data_aug_500x500 import data_aug
import sys
import theano
import os
from PIL import Image
import vgg16.vgg16
from sklearn.metrics import roc_auc_score
import glob
from batch_norms import batch_norm
from shape import ReshapeLayer
from math import floor

import scipy.misc

# Command arguments
script_file_name = sys.argv[0][:-3];
#fold_id = int(sys.argv[1]);
fold_id = 0;

# Parameters
reload_model = True;    # This is for loading an available model to continue training if there exists an available model
with_auginfo = False;
n_time_duplicate = 10;   # Param for duplicating negative data
blob_filter_size = 37;
n_unit_hiddenlayer = 100;
n_binaryclassifier = 1;
#APS = 500;
#PS = 200;
APS = None;
PS = None;
batchsize = 10;
num_epoch = 300;
LearningRate = theano.shared(np.array(1e-3, dtype=np.float32));

if (with_auginfo == True):
    aug_dim = blob.get_feature_size(blob_filter_size);
else:
    aug_dim = 1;

# Input dirs
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt5_mu.pkl';
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt5_sigma.pkl';
#encoder_file = './model_vals_saved/deep_conv_autoencoder_spsparse_alt3_model_1.pkl';
all_data_folder = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/data/vals/';
#train_folder_list = ['/data08/shared/lehhou/nucleus_encoder/data/vals/necrosis_seg_train'];
#train_folder_list = ['/data09/shared/lehhou/theano/nucleus/nucleus_encoder/data/vals/necrosis_segmentation_sep_2_train', \
        #'/data09/shared/lehhou/theano/nucleus/nucleus_encoder/data/vals/necrosis_segmentation_additional_sep_15_train'];
train_folder_list = [];
#test_folder_list = ['/data08/shared/lehhou/nucleus_encoder/data/vals/necrosis_seg_test'];
#test_folder_list = ['/data09/shared/lehhou/theano/nucleus/nucleus_encoder/data/vals/necrosis_segmentation_sep_2_test', \
        #'/data09/shared/lehhou/theano/nucleus/nucleus_encoder/data/vals/necrosis_segmentation_additional_sep_15_test'];
test_folder_list = ['/data08/shared/lehhou/nucleus_encoder/test_data/train_Apr10_old_train/'];
vgg_model_file = 'vgg16/vgg16.pkl';

# Output dirs
model_idx = 270;
classification_model_file = 'model_vals/deep_conv_classification_model_' + script_file_name + '_e{}.pkl'.format(model_idx);
class_result_file = './prediction_result/result-class_' + script_file_name + '_foldid-' + str(fold_id) + '_' + '.pkl';
pred_result_file = './prediction_result/result-pred_' + script_file_name + '_foldid-' + str(fold_id);

mu = None;
sigma = None;


def load_seg_data():
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_train = np.zeros(shape=(0, APS, APS), dtype=np.float32);
    X_test = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_test = np.zeros(shape=(0, APS, APS), dtype=np.float32);

    for train_set in train_folder_list:
        X_tr, y_tr = load_seg_data_folder(train_set);
        X_train = np.concatenate((X_train, X_tr));
        y_train = np.concatenate((y_train, y_tr));

    for test_set in test_folder_list:
        X_ts, y_ts = load_seg_data_folder(test_set);
        X_test = np.concatenate((X_test, X_ts));
        y_test = np.concatenate((y_test, y_ts));

    if (reload_model == False):
        print "Computing mean and std";
        global mu;
        global sigma;
        mu = np.mean(X_train[0::int(floor(X_train.shape[0]/1)), :, :, :].flatten());
        sigma = np.std(X_train[0::int(floor(X_train.shape[0]/1)), :, :, :].flatten());
    print "Mu: ", mu;
    print "Sigma: ", sigma;
    #print "Max X: ", np.amax(X_train), np.amax(X_test);
    #print "Min X: ", np.amin(X_train), np.amin(X_test);
    #print "Avg X: ", np.average(X_train), np.average(X_test);
    print "Shapes: ", X_train.shape, X_test.shape;

    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32), mu, sigma;

def load_seg_data_folder(folder):
    X = np.zeros(shape=(40000, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(40000, APS, APS), dtype=np.float32);

    img_id = 0;
    idx = 0;
    #print get_img_idx(folder, 'image_');
    image_names = read_image_list_file(folder + 'list.txt');
    for img_name in image_names:
    #for img_id in get_img_idx(folder, 'image_'):
        img_id = int(img_name[6:-4]);
        # Load file
        img_png = np.array(Image.open(folder + '/image_' + str(img_id) + '.png').convert('RGB')).transpose();
        mask_png = (np.array(Image.open(folder + '/mask_' + str(img_id) + '.png').convert('L')).transpose() > 0.5);  # we divide by 255 to norm the values to [0, 1]
        X[idx] = img_png;
        y[idx] = mask_png;
        idx += 1;

    X = X[:idx];
    y = y[:idx];

    return X, y;

def read_image_list_file(text_file_path):
    with open(text_file_path) as f:
        content = f.readlines();

    content = [x.strip() for x in content];
    return content;

def get_img_idx(folder, prefix='image_'):
    file_idx = np.zeros(shape=(40000,), dtype=np.int32);
    id = 0;
    print folder + '/' + prefix + '*.png';
    for filename in glob.glob(folder + '/' + prefix + '*.png'):
        file_no_part = filename[(filename.rfind('_') + 1):];
        file_idx[id] = int(file_no_part[:-4]);
        id += 1;

    file_idx = file_idx[:id];
    file_idx = np.sort(file_idx);
    return file_idx;


def load_data(classn, valid_num):
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_train = np.zeros(shape=(0, classn), dtype=np.int32);

    lines = [line.rstrip('\n') for line in open(all_data_folder + '/necrosis_train_test.txt')];
    valid_i = 0;
    for line in lines:
        split_folders = [all_data_folder + s for s in line.split()];
        if valid_i == valid_num:
            X_test, y_test = load_data_split(classn, split_folders, False);
        else:
            X_split, y_split = load_data_split(classn, split_folders, False);
            X_train = np.concatenate((X_train, X_split));
            y_train = np.concatenate((y_train, y_split));
        valid_i += 1;

    print "Data Loaded", X_train.shape, y_train.shape, X_test.shape, y_test.shape;
    return X_train, y_train, X_test, y_test;


def load_data_split(classn, folders, isduplicate):
    X = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(0, classn), dtype=np.int32);
    for folder in folders:
        X_split, y_split = load_data_folder(classn, folder, isduplicate);
        X = np.concatenate((X, X_split));
        y = np.concatenate((y, y_split));
    return X, y;

def load_data_folder(classn, folder, isduplicate):
    X = np.zeros(shape=(100000, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(100000, classn), dtype=np.int32);

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
        if ((isduplicate == True) and (lab == 0)):
            for i in range(n_time_duplicate):
                X[nline], y[nline] = png, lab;
                nline += 1;


    X = X[0:nline];
    y = y[0:nline];
    return X, y;

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
            excerpt = indices[start_idx: start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt], augs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx + batchsize: len(inputs)];
        else:
            excerpt = slice(start_idx + batchsize, len(inputs));
        yield inputs[excerpt], augs[excerpt], targets[excerpt];

def build_classfication_model_from_vgg16 ():
    layer_list, vgg_whole, vgg_input_var = vgg16.vgg16.build_model();

    vgg_cut = layer_list['fc8'];

    aug_var = theano.tensor.matrix('aug_var');
    aug_layer = lasagne.layers.InputLayer(shape=(None, aug_dim), input_var = aug_var);

    layer_list['aggregate_layer'] = lasagne.layers.ConcatLayer([vgg_cut,aug_layer], axis = 1);

    layer_list['last_sigmoid'] = lasagne.layers.DenseLayer(incoming=layer_list['aggregate_layer'], num_units=n_binaryclassifier, nonlinearity=lasagne.nonlinearities.sigmoid);
    network = layer_list['last_sigmoid'];

    latter_param = [layer_list['last_sigmoid'].W, layer_list['last_sigmoid'].b];
    all_param = lasagne.layers.get_all_params(network, trainable=True);

    return network, vgg_whole, layer_list, all_param, latter_param, vgg_input_var, aug_var;


def build_deconv_network_temp():
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

    network = ReshapeLayer(net['seg_score'], ([0], -1));
    output_var = lasagne.layers.get_output(network);
    all_param = lasagne.layers.get_all_params(network, trainable=True);

    return network, input_var, output_var, all_param;




def build_deconv_network():
    input_var = theano.tensor.tensor4('input_var');

    net = {};
    net['input'] = layers.InputLayer(shape=(None, 3, PS, PS), input_var=input_var);


    # Encoding part
    net['conv1_1'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['input'], 64,   filter_size=(3,3), stride=1, pad=1)));
    net['conv1_2'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv1_1'], 64,   filter_size=(3,3), stride=1, pad=1)));
    net['pool1'] = layers.Pool2DLayer(net['conv1_2'], pool_size=(2,2), stride=2, mode='max');

    net['conv2_1'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['pool1'], 128,   filter_size=(3,3), stride=1, pad=1)));
    net['conv2_2'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv2_1'], 128,   filter_size=(3,3), stride=1, pad=1)));
    net['pool2'] = layers.Pool2DLayer(net['conv2_2'], pool_size=(2,2), stride=2, mode='max');

    net['conv3_1'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['pool2'], 256,   filter_size=(3,3), stride=1, pad=1)));
    net['conv3_2'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv3_1'], 256,   filter_size=(3,3), stride=1, pad=1)));
    net['conv3_3'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv3_2'], 256,   filter_size=(3,3), stride=1, pad=1)));
    net['pool3'] = layers.Pool2DLayer(net['conv3_3'], pool_size=(2,2), stride=2, mode='max');

    net['conv4_1'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['pool3'], 512,   filter_size=(3,3), stride=1, pad=1)));
    net['conv4_2'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv4_1'], 512,   filter_size=(3,3), stride=1, pad=1)));
    net['conv4_3'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv4_2'], 512,   filter_size=(3,3), stride=1, pad=1)));
    net['pool4'] = layers.Pool2DLayer(net['conv4_3'], pool_size=(2,2), stride=2, mode='max');

    net['conv5_1'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['pool4'], 512,   filter_size=(3,3), stride=1, pad=1)));
    net['conv5_2'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv5_1'], 512,   filter_size=(3,3), stride=1, pad=1)));
    net['conv5_3'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['conv5_2'], 512,   filter_size=(3,3), stride=1, pad=1)));
    net['pool5'] = layers.Pool2DLayer(net['conv5_3'], pool_size=(2,2), stride=2, mode='max');

    net['fc6'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['pool5'], 4096,   filter_size=(7,7), stride=1, pad='same')));

    # fc7 is the encoding layer
    net['fc7'] = layers.NonlinearityLayer(batch_norm(layers.Conv2DLayer(net['fc6'], 4096,   filter_size=(1,1), stride=1, pad='same')));

    # Decoding part
    net['fc6_deconv'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['fc7'], 512, filter_size=(7,7), stride=1, crop='same')));
    net['unpool5'] = layers.InverseLayer(net['fc6_deconv'], net['pool5']);

    net['deconv5_1'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['unpool5'], 512, filter_size=(3,3), stride=1, crop='same')));
    net['deconv5_2'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv5_1'], 512, filter_size=(3,3), stride=1, crop='same')));
    net['deconv5_3'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv5_2'], 512, filter_size=(3,3), stride=1, crop='same')));
    net['unpool4'] = layers.InverseLayer(net['deconv5_3'], net['pool4']);

    net['deconv4_1'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['unpool4'], 512, filter_size=(3,3), stride=1, crop='same')));
    net['deconv4_2'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv4_1'], 512, filter_size=(3,3), stride=1, crop='same')));
    net['deconv4_3'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv4_2'], 256, filter_size=(3,3), stride=1, crop='same')));
    net['unpool3'] = layers.InverseLayer(net['deconv4_3'], net['pool3']);

    net['deconv3_1'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['unpool3'], 256, filter_size=(3,3), stride=1, crop='same')));
    net['deconv3_2'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv3_1'], 256, filter_size=(3,3), stride=1, crop='same')));
    net['deconv3_3'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv3_2'], 128, filter_size=(3,3), stride=1, crop='same')));
    net['unpool2'] = layers.InverseLayer(net['deconv3_3'], net['pool2']);

    net['deconv2_1'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['unpool2'], 128, filter_size=(3,3), stride=1, crop='same')));
    net['deconv2_2'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv2_1'], 64, filter_size=(3,3), stride=1, crop='same')));
    net['unpool1'] = layers.InverseLayer(net['deconv2_2'], net['pool1']);

    net['deconv1_1'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['unpool1'], 64, filter_size=(3,3), stride=1, crop='same')));
    net['deconv1_2'] = layers.NonlinearityLayer(batch_norm(layers.Deconv2DLayer(net['deconv1_1'], 64, filter_size=(3,3), stride=1, crop='same')));


    # Segmentation layer
    net['seg_score'] = layers.Deconv2DLayer(net['deconv1_2'], 1, filter_size=(1,1), stride=1, crop='same', nonlinearity=lasagne.nonlinearities.sigmoid);

    network = ReshapeLayer(net['seg_score'], ([0], -1));
    output_var = lasagne.layers.get_output(network);
    all_param = lasagne.layers.get_all_params(network, trainable=True);

    return network, input_var, output_var, all_param;





def load_model_value(network, model_file):
    loaded_var = pickle.load(open(model_file, 'rb'));
    global mu;
    global sigma;
    mu = loaded_var[0];
    sigma = loaded_var[1];

    param_values = loaded_var[2];
    '''
    param_set = lasagne.layers.get_all_params(network);
    for param, value in zip(param_set, param_values):
        param.set_value(value);
    '''
    lasagne.layers.set_all_param_values(network, param_values);

def build_training_function(network, param_set, input_var, target_var):
    prediction_var_train = lasagne.layers.get_output(network, deterministic=False);
    prediction_var_test = lasagne.layers.get_output(network, deterministic=True);
    loss_train = lasagne.objectives.squared_error(prediction_var_train, target_var).mean();
    loss_test = lasagne.objectives.squared_error(prediction_var_test, target_var).mean();

    # training function
    updates = lasagne.updates.nesterov_momentum(loss_train, param_set, learning_rate=LearningRate, momentum=0.985);
    train_func = theano.function([input_var, target_var], loss_train, updates=updates);

    test_func = theano.function([input_var, target_var], [loss_test, prediction_var_test]);

    print("finish building training function");
    return train_func, test_func;

'''
def build_testing_function(network, input_var, aug_var):
    target_var = theano.iscalar('target_var');
    prediction_var = lasagne.layers.get_output(network, deterministic=True);
    error = lasagne.objectives.binary_crossentropy(prediction_var, target_var);

    test_func = theano.function([input_var, aug_var, target_var], error);
    return test_func;
'''

def cross_validation(X_test, y_test, loaded_mu, loaded_sigma, train_func, test_func):
    # load data
    #X_train, y_train, X_test, y_test, computed_mu, computed_sigma = load_seg_data();
    #y_train = y_train.reshape(y_train.shape[0], -1);
    #y_test  = y_test.reshape(y_test.shape[0], -1);
    global mu;
    global sigma;
    mu = loaded_mu;
    sigma = loaded_sigma;

    # Generating augmenting data
    """
    if (with_auginfo == True):
        print "Begin extracting blob features...";
        X_train_aug = data_aug(X_train, mu, sigma);
        X_test_aug = data_aug(X_test, mu, sigma);
        a_train = generate_aug_data(X_train_aug);
        a_test = generate_aug_data(X_test_aug);
        print "Finish extracting blob features...";
    else:
        a_train = np.zeros(shape=(X_train.shape[0], n_binaryclassifier), dtype=np.float32);
        a_test = np.zeros(shape=(X_test.shape[0], n_binaryclassifier), dtype=np.float32);
    """

    #a_train = np.zeros(shape=(X_train.shape[0], 1), dtype=np.float32);
    a_test = np.zeros(shape=(X_test.shape[0], 1), dtype=np.float32);

    # do training
    #param_values = exc_train(train_func, test_func, X_train, a_train, y_train, X_test, a_test, y_test, network, param_set);

    # do testing
    #opt_network, _, _, _, _, opt_input_var, opt_aug_var = build_classfication_model_from_ae();
    #lasagne.layers.set_all_param_values(opt_network, param_values);
    image_array, groundtruth_array, prediction_array = exc_test(test_func, X_test, a_test, y_test);
    #write_to_image(image_array, groundtruth_array, prediction_array)

    return image_array, groundtruth_array, prediction_array;

def write_to_image(img, gt, pred):
    # abc
    print "write to image ", pred.shape;
    for idx in range(pred.shape[0]):
        written = img[idx].transpose();
        filename = './necrosis_test_img/image_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);

        print "gt shape", gt[idx].shape;
        written = np.reshape(gt[idx], (APS, APS)).transpose();
        filename = './necrosis_test_img/gt_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);

        written = np.reshape(pred[idx], (APS, APS)).transpose();
        filename = './necrosis_test_img/pred_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);


def write_to_image_temp(img, pred, idx):
    # abc
    print "write to image ", pred.shape;
    if (len(img.shape) == 4):
        written = img[0].transpose();
    else:
        written = img.transpose();
    filename = './necrosis_test_img/image_' + str(idx) + '.png';
    print "aaa ", written.shape, pred.shape;
    scipy.misc.imsave(filename, written);

    if (len(pred.shape) == 3):
        written = pred[0].transpose();
    else:
        written = pred.transpose();
    filename = './necrosis_test_img/pred_' + str(idx) + '.png';
    scipy.misc.imsave(filename, written);


def exc_train(train_func, test_func, X_train, a_train, y_train, X_test, a_test, y_test, network, param_set):
    print("Start training...");
    print "Learning Rate: ", LearningRate.get_value();

    min_loss = 1e4;
    temp_boolean = True;
    for epoch in range(num_epoch):
        i_batch = 0;
        total_loss = 0;
        start_time = time.time();
        for batch in iterate_minibatches(X_train, a_train, y_train, batchsize, shuffle = True):
            input_real, aug_real, target_real = batch;


            input_real, target_real = data_aug(input_real, target_real, mu, sigma, APS, PS);

            if (temp_boolean == True):
                print "size of img ", input_real.shape;
                print "size of target ", target_real.shape;
                temp_boolean = False;

            target_real = target_real.reshape(target_real.shape[0], -1);
            loss = train_func(input_real, target_real);
            total_loss += loss;

            i_batch += 1;

        print "finish one epoch";
        epoch_loss = total_loss/i_batch;
        print ("{:.4f}\tin {:.1f} (s)".format(epoch_loss, time.time()-start_time));

        if epoch == 40:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 100:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 200:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));

        if (epoch % 10 == 0):
            # save model
            print "save model";
            save_model(network, classification_model_file.format(epoch));

        # Remember to change the below line to your preferred number
        if (epoch % 5 == 0):
            # validate on test data
            aug_img_array, groundtruth_array, prediction_array, test_loss, tr_pos, fl_pos, tr_neg, fl_neg, auc = exc_test(test_func, X_test, a_test, y_test);
            print("Validation:\t\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.3f}".format(test_loss, tr_pos, fl_pos, tr_neg, fl_neg, auc));

            if (epoch % 10 == 0):
                # save the prediction result
                #pickle.dump(prediction_array, open(pred_result_file, 'w'));
                timestr = time.strftime("%Y%m%d-%H%M%S");
                pred_file_timefmt = pred_result_file + '_' + timestr;
                np.save(pred_file_timefmt + '_RGB', aug_img_array);
                np.save(pred_file_timefmt + '_mask', groundtruth_array);
                np.save(pred_file_timefmt + '_pred', prediction_array);
                #np.savez(pred_file_timefmt, aug_img_array, groundtruth_array, prediction_array);
                print "save result...";

            #aug_img_array, groundtruth_array, prediction_array, test_loss, tr_pos, fl_pos, tr_neg, fl_neg, auc = exc_test(test_func, X_train, a_train, y_train);
            #print("Valid on Train:\t\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.3f}".format(test_loss, tr_pos, fl_pos, tr_neg, fl_neg, auc));

    return lasagne.layers.get_all_param_values(network);

def exc_test(test_func, X_test, a_test, y_test):
    print("Start testing...");

    #i_draw_arr = [0,150,300];
    #j_draw_arr = [0,150,300];
    #i_draw_arr = [0, (APS - PS)/2, APS - PS];
    #j_draw_arr = [0, (APS - PS)/2, APS - PS];
    i_draw_arr = [0, APS - PS];
    j_draw_arr = [0, APS - PS];
    print "i_draw, j_draw ", i_draw_arr, j_draw_arr;

    total_error = 0;
    i_batch = 0;
    i_line = 0;
    """
    image_array = np.empty(shape=(len(i_draw_arr)*len(j_draw_arr)*X_test.shape[0], 3, PS, PS), dtype=np.float32);
    prediction_array = np.empty(shape=(len(i_draw_arr)*len(j_draw_arr)*X_test.shape[0], PS**2), dtype=np.float32);
    groundtruth_array = np.empty(shape=(len(i_draw_arr)*len(j_draw_arr)*X_test.shape[0], PS**2), dtype=np.int32);
    """

    image_array = np.empty(shape=(X_test.shape[0], 3, APS, APS), dtype=np.float32);
    prediction_array = np.empty(shape=(X_test.shape[0], APS, APS), dtype=np.float32);
    groundtruth_array = np.empty(shape=(X_test.shape[0], APS, APS), dtype=np.int32);

    class_array = np.empty(shape=(len(i_draw_arr)*len(j_draw_arr)*X_test.shape[0], PS**2), dtype=np.float32);

    tr_pos = 0;
    fl_pos = 0;
    fl_neg = 0;
    tr_neg = 0;

    # Debug
    total_pixels = 0;

    begin_img_idx = 0;
    for batch in iterate_minibatches(X_test, a_test, y_test, batchsize, shuffle = False):
        #print begin_img_idx;
        input_real_org, aug_real_org, target_real_org = batch;
        #input_real = data_aug(input_real, mu, sigma);

        image_patch = np.zeros(shape=(input_real_org.shape[0], 3, APS, APS), dtype=np.float32);
        prediction_patch = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);
        groundtruth_patch = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);

        weight_2d = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);
        mask_2d = np.ones(shape=(input_real_org.shape[0], PS, PS), dtype=np.float32);
        weight_3d = np.zeros(shape=(input_real_org.shape[0], 3, APS, APS), dtype=np.float32);
        mask_3d = np.ones(shape=(input_real_org.shape[0], 3, PS, PS), dtype=np.float32);

        #print "image_patch, mask_3d ", image_patch.shape, mask_3d.shape;

        temp_idx = 0;
        for i_draw in i_draw_arr:
            for j_draw in j_draw_arr:

                input_real, target_real = data_aug(X=input_real_org, Y=target_real_org, mu=mu, sigma=sigma, deterministic=True, idraw=i_draw, jdraw=j_draw, APS=APS, PS=PS);
                #y_test_flattened = target_real.reshape(-1, 1);

                target_real = target_real.reshape(target_real.shape[0], -1);
                error, prediction_real = test_func(input_real, target_real);

                class_res = from_pred_to_class(prediction_real);
                class_res_flattened = class_res.reshape(-1, 1);
                """
                image_array[i_line:i_line+len(prediction_real)] = input_real;
                prediction_array[i_line:i_line+len(prediction_real)] = prediction_real;
                groundtruth_array[i_line:i_line+len(prediction_real)] = target_real;
                """
                class_array[i_line:i_line+len(prediction_real)] = class_res;

                image_patch[:, :,i_draw:i_draw+PS, j_draw:j_draw+PS] += input_real;
                #print prediction_real.shape;
                prediction_patch[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += np.reshape(prediction_real, (-1, PS, PS));
                groundtruth_patch[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += np.reshape(target_real, (-1, PS, PS));
                weight_2d[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += mask_2d;
                weight_3d[:, :, i_draw:i_draw+PS, j_draw:j_draw+PS] += mask_3d;

                total_error += error;
                i_batch += 1;
                i_line += len(prediction_real);

                #write_to_image_temp(input_real, np.reshape(prediction_real, (-1, PS, PS)), temp_idx);

                #print "y_test_flattened, class_res_flattened: ", y_test_flattened.shape, class_res_flattened.shape;
                """
                pred_pos_id_comp = np.where(class_res_flattened == 1);
                pred_neg_id_comp = np.where(class_res_flattened == 0);

                tr_pos_comp = y_test_flattened[pred_pos_id_comp[0]].sum();
                fl_pos_comp = len(pred_pos_id_comp[0]) - tr_pos_comp;
                fl_neg_comp = y_test_flattened[pred_neg_id_comp[0]].sum();
                tr_neg_comp = len(pred_neg_id_comp[0]) - fl_neg_comp;

                tr_pos += tr_pos_comp;
                fl_pos += fl_pos_comp;
                fl_neg += fl_neg_comp;
                tr_neg += tr_neg_comp;

                # Debug
                total_pixels += y_test_flattened.shape[0];
                """
                temp_idx += 1;

        image_patch = np.divide(image_patch, weight_3d);
        prediction_patch = np.divide(prediction_patch, weight_2d);
        groundtruth_patch = np.divide(groundtruth_patch, weight_2d);

        image_array[begin_img_idx:begin_img_idx+input_real_org.shape[0]] = image_patch;
        prediction_array[begin_img_idx:begin_img_idx+input_real_org.shape[0]] = prediction_patch;
        groundtruth_array[begin_img_idx:begin_img_idx+input_real_org.shape[0]] = groundtruth_patch;

        begin_img_idx += input_real_org.shape[0];


    """
    tr_pos = float(tr_pos) / (PS**2);
    fl_pos = float(fl_pos) / (PS**2);
    fl_neg = float(fl_neg) / (PS**2);
    tr_neg = float(tr_neg) / (PS**2);

    print "Sum of components: ", (tr_pos + fl_pos + fl_neg + tr_neg);
    print "Total pixels: ", total_pixels;

    total_error = total_error / i_batch;

    #pred_pos_id = np.where(class_array == 1);
    #pred_neg_id = np.where(class_array == 0);

    #tr_pos = y_test[pred_pos_id[0]].sum();
    #fl_pos = len(pred_pos_id[0]) - tr_pos;
    #fl_neg = y_test[pred_neg_id[0]].sum();
    #tr_neg = len(pred_neg_id[0]) - fl_neg;

    # compute AUC
    #auc = roc_auc_score(groundtruth_array, prediction_array.reshape(-1, 1));
    """

    print "image, gt, pred: ", image_array.shape, groundtruth_array.shape, prediction_array.shape;
    #return image_array, groundtruth_array, prediction_array, total_error, tr_pos, fl_pos, tr_neg, fl_neg;
    return image_array, groundtruth_array, prediction_array;


def from_pred_to_class(pred):
    class_res = np.copy(pred);
    class_res = (class_res >= 0.5).astype(np.int32);
    return class_res;


def save_model(out_layer, file_path):
    all_param_values = lasagne.layers.get_all_param_values(out_layer);
    #all_param_values = [p.get_value() for p in all_params];
    #pickle.dump(all_param_values, open(file_path, 'w'));
    with open(file_path, 'w') as f:
        pickle.dump([mu, sigma, all_param_values], f);



def necrosis_predict(X_test, Y_test, loaded_mu, loaded_sigma, param_values, loaded_APS, loaded_PS):

    # attach additional layers for classification
    #network, encoder_layer, whole_autoencoder, latter_param, all_param, input_var, aug_var = build_classfication_model_from_ae();
    #network, vgg_whole, layer_list, all_param, latter_param, input_var, aug_var = build_classfication_model_from_vgg16();
    network, input_var, output_var, all_param = build_deconv_network_temp();

    # load param values of the encoder
    """
    if (reload_model == True):
        print classification_model_file;
        print "Loading ", classification_model_file;
        if (os.path.isfile(classification_model_file) == True):
            # load model of the whole classificaiton network
            load_model_value(network, classification_model_file);
            print "finish reloading model";
        else:
            print "No available model to load";
            return;
    #else:
        # load model of only the autoencoder part
        #load_model_value(vgg_whole, vgg_model_file);

        #model = pickle.load(open(vgg_model_file));
        #lasagne.layers.set_all_param_values(vgg_whole, model['param values']);
    """

    global mu;
    global sigma;
    global APS;
    global PS;
    mu = loaded_mu;
    sigma = loaded_sigma;
    lasagne.layers.set_all_param_values(network, param_values);
    APS = loaded_APS;
    PS = loaded_PS;

    # build train function
    # you can change the set of params to training by switching between "latter_param" and "all_param" in the command below
    target_var = theano.tensor.imatrix('target_var');
    train_func, test_func = build_training_function(network, all_param, input_var, target_var);

    # run training  (X_test, y_test, loaed_mu, loaded_sigma, train_func, test_func)
    image_array, groundtruth_array, prediction_array = cross_validation(X_test, Y_test, mu, sigma, train_func, test_func);

    print "DONE!";

    return image_array, groundtruth_array, prediction_array;

