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
import PIL
from PIL import Image
import vgg16.vgg16
from sklearn.metrics import roc_auc_score
import glob
from batch_norms import batch_norm
from shape import ReshapeLayer
from math import floor
from pred_necrosis_batch import necrosis_predict

import scipy.misc
import os.path


def load_seg_data(train_folder_list, test_folder_list, APS):
    X_train = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_train = np.zeros(shape=(0, APS, APS), dtype=np.float32);
    X_test = np.zeros(shape=(0, 3, APS, APS), dtype=np.float32);
    y_test = np.zeros(shape=(0, APS, APS), dtype=np.float32);
    image_name_train = [];
    image_name_test = [];

    for train_set in train_folder_list:
        X_tr, y_tr, image_name_train = load_seg_data_folder(train_set, APS);
        X_train = np.concatenate((X_train, X_tr));
        y_train = np.concatenate((y_train, y_tr));

    for test_set in test_folder_list:
        X_ts, y_ts, image_name_test = load_seg_data_folder(test_set, APS);
        X_test = np.concatenate((X_test, X_ts));
        y_test = np.concatenate((y_test, y_ts));

    print "Shapes: ", X_train.shape, X_test.shape;
    return X_train, y_train.astype(np.int32), image_name_train, X_test, y_test.astype(np.int32), image_name_test;


def load_seg_data_folder(folder, APS):
    X = np.zeros(shape=(40000, 3, APS, APS), dtype=np.float32);
    y = np.zeros(shape=(40000, APS, APS), dtype=np.float32);

    idx = 0;
    image_names = read_image_list_file(folder + '/list.txt');
    for img_name in image_names:
        # Load file
        loaded_png = Image.open(folder + '/' + img_name + '.png');
        resized_png = loaded_png.resize((APS, APS), PIL.Image.ANTIALIAS);
        img_png = np.array(resized_png.convert('RGB')).transpose();
        mask_png = np.zeros(shape=(1, APS, APS), dtype=np.float32);
        X[idx] = img_png;
        y[idx] = mask_png;
        idx += 1;

    X = X[:idx];
    y = y[:idx];

    return X, y, image_names;

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

def load_model_value(model_file):
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
    return mu, sigma, param_values;
    #lasagne.layers.set_all_param_values(network, param_values);

def write_to_image(img, pred):
    # abc
    print "write to image ", pred.shape;
    for idx in range(pred.shape[0]):
        written = img[idx].transpose();
        filename = './necrosis_test_img_big/image_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);

        """
        print "gt shape", gt[idx].shape;
        written = np.reshape(gt[idx], (APS, APS)).transpose();
        filename = './necrosis_test_img_big/gt_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);
        """

        written = pred[idx].transpose();
        filename = './necrosis_test_img_big/pred_' + str(idx) + '.png';
        scipy.misc.imsave(filename, written);

def predict_slide(slide_folder, mu, sigma, param_values, heatmap_folder):
    # Get list of image files
    list_file_path = slide_folder + '/list.txt';
    img_name_list = [];
    if (os.path.isfile(list_file_path) == False):
        print "list file not avaible, produce one";
        f = open(list_file_path, 'w')
        path_list = glob.glob(slide_folder + '/*.png');
        for img_path in path_list:
            base=os.path.basename(img_path);
            img_name = os.path.splitext(base)[0];
            f.write(img_name + '\n');
        f.close();

    with open(list_file_path) as f:
        content = f.readlines();
    img_name_list = [x.strip() for x in content];

    # Analyze APS, PS
    APS = 333;
    PS = 200;

    """
    # Load model file (should be put in upper level later)
    print ('Load model file...');
    mu, sigma, param_values = load_model_value(classification_model_file);
    print ('Finish loading model file');
    """

    # Load testing data
    print ('Load testing data...');
    X_train, y_train, image_name_train, X_test, y_test, image_name_test = load_seg_data([], [slide_folder], APS);
    print ('Finish loading testing data');

    # Do prediction
    print ('Do prediction...');
    image_array, groundtruth_array, prediction_array = necrosis_predict(X_test, y_test, mu, sigma, param_values, APS, PS);
    print "Output shape: image, groundtruth, prediction ", image_array.shape, groundtruth_array.shape, prediction_array.shape;

    # Write to image to test - need to be remove in the final code ???
    #write_to_image(image_array, prediction_array);

    # Divide the grid 10x10
    print ('Convert to lymphocyte size...');

    parent_path, slide_name = os.path.split(slide_folder);
    heatmap_path = heatmap_folder + '/patch-level-necrosis.txt';
    f_res = open(heatmap_path, 'w')
    for idx, big_patch_name in enumerate(image_name_test):
        parts = big_patch_name.split('_');
        root_x = int(parts[0]);
        root_y = int(parts[1]);
        abs_size = int(parts[2]);
        big_patch = prediction_array[idx];

        loc_arr = [x * 0.1 + 0.05 for x in range(0, 10)];

        for x_idx, abs_x in enumerate(xrange(0,300,33)):
            for y_idx, abs_y in enumerate(xrange(0,300,33)):
                real_x_loc = int(loc_arr[x_idx] * abs_size + root_x);
                real_y_loc = int(loc_arr[y_idx] * abs_size + root_y);
                avg_val = np.average(big_patch[abs_x : abs_x + 33, abs_y : abs_y + 33]);
                f_res.write("{0} {1} {2}\n".format(real_x_loc, real_y_loc, avg_val));

    f_res.close();

