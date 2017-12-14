import os
import sys
import pickle
import numpy as np
import lasagne
import time
import theano
import glob
import scipy.misc

from math import floor
from PIL import Image
from lasagne import layers
from lasagne.nonlinearities import sigmoid, leaky_rectify
from sklearn.metrics import roc_auc_score
from data_aug_500x500 import data_aug
from batch_norms import batch_norm
from shape import ReshapeLayer

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
batchsize = 2;
num_epoch = 300;
LearningRate = theano.shared(np.array(1e-3, dtype=np.float32));

if (with_auginfo == True):
    aug_dim = blob.get_feature_size(blob_filter_size);
else:
    aug_dim = 1;

# Input dirs
filename_mu = 'model_vals/deep_conv_autoencoder_spsparse_alt5_mu.pkl';
filename_sigma = 'model_vals/deep_conv_autoencoder_spsparse_alt5_sigma.pkl';
train_folder_list = [];
test_folder_list = [];

# Output dirs
model_idx = None;
classification_model_file = None;
class_result_file = './prediction_result/result-class_' + script_file_name + '_foldid-' + str(fold_id) + '_' + '.pkl';
pred_result_file = './prediction_result/result-pred_' + script_file_name + '_foldid-' + str(fold_id);

def load_seg_data(train_folder_list, test_folder_list, APS, PS):
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_train = np.zeros(shape=(0, APS, APS), dtype=np.float32);
    X_test = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_test = np.zeros(shape=(0, APS, APS), dtype=np.float32);

    for train_set in train_folder_list:
        X_tr, y_tr = load_seg_data_folder(train_set, APS, PS);
        X_train = np.concatenate((X_train, X_tr));
        y_train = np.concatenate((y_train, y_tr));

    for test_set in test_folder_list:
        X_ts, y_ts = load_seg_data_folder(test_set, APS, PS);
        X_test = np.concatenate((X_test, X_ts));
        y_test = np.concatenate((y_test, y_ts));

    print "Computing mean and std";
    print "Data Shapes: ", X_train.shape, X_test.shape;
    mu = np.mean(X_train[0::int(floor(X_train.shape[0]/1)), :, :, :].flatten());
    sigma = np.std(X_train[0::int(floor(X_train.shape[0]/1)), :, :, :].flatten());
    print "Data Shapes: ", X_train.shape, y_train.shape;

    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32), mu, sigma;

def load_seg_data_folder(folder, APS, PS):
    X = np.zeros(shape=(40000, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(40000, APS, APS), dtype=np.float32);

    img_id = 0;
    idx = 0;
    print get_img_idx(folder, 'image_');
    for img_id in get_img_idx(folder, 'image_'):
        #img_id = int(img_name[6:-4]);
        # Load file
        img_png = np.array(Image.open(folder + '/image_' + str(img_id) + '.png').convert('RGB')).transpose();
        mask_png = (np.array(Image.open(folder + '/mask_' + str(img_id) + '.png').convert('L')).transpose() > 0.5);  # we divide by 255 to norm the values to [0, 1]
        X[idx] = img_png;
        y[idx] = mask_png;
        idx += 1;

    X = X[:idx];
    y = y[:idx];

    return X, y;


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
            excerpt = indices[start_idx: start_idx + batchsize];
        else:
            excerpt = slice(start_idx, start_idx + batchsize);
        yield inputs[excerpt], targets[excerpt];
    if start_idx < len(inputs) - batchsize:
        if shuffle:
            excerpt = indices[start_idx + batchsize: len(inputs)];
        else:
            excerpt = slice(start_idx + batchsize, len(inputs));
        yield inputs[excerpt], targets[excerpt];


def build_deconv_network(APS, PS):
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

def load_model_value(network, model_file):
    if (os.path.isfile(model_file) == True):
        loaded_var = pickle.load(open(model_file, 'rb'));
        mu = loaded_var[0];
        sigma = loaded_var[1];

        param_values = loaded_var[2];
        '''
        param_set = lasagne.layers.get_all_params(network);
        for param, value in zip(param_set, param_values):
            param.set_value(value);
        '''
        lasagne.layers.set_all_param_values(network, param_values);

        return mu, sigma;
    else:
        return None, None;

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


def exc_train(train_func, test_func, mu, sigma, X_train, y_train, X_test, y_test, network, APS, PS, classification_model_file, epochno_model_save, epochno_validate):
    print("Start training...");
    print "Learning Rate: ", LearningRate.get_value();

    min_loss = 1e4;
    for epoch in range(num_epoch):
        i_batch = 0;
        total_loss = 0;
        start_time = time.time();
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle = True):
            input_real, target_real = batch;
            input_real, target_real = data_aug(input_real, target_real, mu, sigma, deterministic=False, idraw=-1, jdraw=-1, APS=APS, PS=PS);
            target_real = target_real.reshape(target_real.shape[0], -1);

            loss = train_func(input_real, target_real);
            total_loss += loss;

            i_batch += 1;

        print "finish one epoch";
        epoch_loss = total_loss/i_batch;
        print ("Epoch loss {:.4f}\tTime {:.1f} (s)".format(epoch_loss, time.time()-start_time));

        if epoch == 40:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 100:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));
        if epoch == 200:
            LearningRate.set_value(np.float32(0.10*LearningRate.get_value()));

        if (epoch % epochno_model_save == 0):
            # save model
            print "save model";
            save_model(network, mu, sigma, classification_model_file.format(epoch));

        # Remember to change the below line to your preferred number
        if (epoch % epochno_validate == 0):
            # validate on test data
            aug_img_array, groundtruth_array, prediction_array, test_loss, tr_pos, fl_pos, tr_neg, fl_neg, auc = exc_test_during_train(test_func, X_test, y_test, mu, sigma, APS, PS);
            print("Validation:\t\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.3f}".format(test_loss, tr_pos, fl_pos, tr_neg, fl_neg, auc));


    return lasagne.layers.get_all_param_values(network);

def exc_test_during_train(test_func, X_test, y_test, mu, sigma, APS, PS):
    print("Start testing...");

    i_draw_arr = [-1];
    j_draw_arr = [-1];

    total_error = 0;
    i_batch = 0;
    i_line = 0;
    prediction_array = np.empty(shape=(len(i_draw_arr)*len(j_draw_arr)*X_test.shape[0], PS**2), dtype=np.float32);
    class_array = np.empty(shape=(len(i_draw_arr)*len(j_draw_arr)*X_test.shape[0], PS**2), dtype=np.float32);
    groundtruth_array = np.empty(shape=(0, 1), dtype=np.int32);
    aug_img_array = np.empty(shape=(0, 3, PS, PS), dtype=np.float32);
    tr_pos = 0;
    fl_pos = 0;
    fl_neg = 0;
    tr_neg = 0;

    for batch in iterate_minibatches(X_test, y_test, batchsize, shuffle = False):
        input_real_org, target_real_org = batch;
        #input_real = data_aug(input_real, mu, sigma);

        for i_draw in i_draw_arr:
            for j_draw in j_draw_arr:

                input_real, target_real = data_aug(input_real_org, target_real_org, mu, sigma, deterministic=True, idraw=i_draw, jdraw=j_draw, APS=APS, PS=PS);
                y_test_flattened = target_real.reshape(-1, 1);
                groundtruth_array = np.concatenate((groundtruth_array, y_test_flattened));
                aug_img_array = np.concatenate((aug_img_array, input_real));

                target_real = target_real.reshape(target_real.shape[0], -1);
                error, prediction_real = test_func(input_real, target_real);

                class_res = from_pred_to_class(prediction_real);
                class_res_flattened = class_res.reshape(-1, 1);
                prediction_array[i_line:i_line+len(prediction_real)] = prediction_real;
                class_array[i_line:i_line+len(prediction_real)] = class_res;

                total_error += error;
                i_batch += 1;
                i_line += len(prediction_real);

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

    tr_pos = float(tr_pos) / (PS**2);
    fl_pos = float(fl_pos) / (PS**2);
    fl_neg = float(fl_neg) / (PS**2);
    tr_neg = float(tr_neg) / (PS**2);

    total_error = total_error / i_batch;

    # compute AUC
    auc = roc_auc_score(groundtruth_array, prediction_array.reshape(-1, 1));

    return aug_img_array, groundtruth_array, prediction_array, total_error, tr_pos, fl_pos, tr_neg, fl_neg, auc;


def from_pred_to_class(pred):
    class_res = np.copy(pred);
    class_res = (class_res >= 0.5).astype(np.int32);
    return class_res;


def save_model(out_layer, mu, sigma, file_path):
    all_param_values = lasagne.layers.get_all_param_values(out_layer);
    #all_param_values = [p.get_value() for p in all_params];
    #pickle.dump(all_param_values, open(file_path, 'w'));
    with open(file_path, 'w') as f:
        pickle.dump([mu, sigma, all_param_values], f);


# existing_model: the path to the existing model, if equals "None", auto-initialize the network
# epochno_model_save: #epoch to save the model
# epochno_validate:   #epoch to test on validate set
def necrosis_train(existing_model, save_model_path, training_folder_list, validation_folder_list, APS, PS, epochno_model_save, epochno_validate):
    # Initialize variables
    classification_model_file = save_model_path + "_e{}.pkl";
    # Initialize the model
    network, input_var, output_var, all_param = build_deconv_network(APS, PS);
    print "Finish build network structure";

    # load data
    X_train, y_train, X_test, y_test, computed_mu, computed_sigma = load_seg_data(training_folder_list, validation_folder_list, APS, PS);
    mu = computed_mu;
    sigma = computed_sigma;
    print "Finish loading data: mu, sigma = ", mu, sigma;

    # Check and load existing model
    if (existing_model != None):
        # Keep training
        mu, sigma = load_model_value(network, existing_model);
        if (mu != None):
            print "Succcessfully loading existing model";
        else:
            print "Fail to load existing model";
            return;

    # Cross validation
    target_var = theano.tensor.imatrix('target_var');
    train_func, test_func = build_training_function(network, all_param, input_var, target_var);

    # do training
    param_values = exc_train(train_func, test_func, mu, sigma, X_train, y_train, X_test, y_test, network, APS, PS, classification_model_file, epochno_model_save, epochno_validate);

