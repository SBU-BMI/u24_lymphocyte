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
from predict_slide import predict_slide

import scipy.misc
import os.path

model_idx = 270;
classification_model_file = 'model_vals/deep_conv_classification_model_deep_segmentation_deconv_necrosis_alt2_e{}.pkl'.format(model_idx);

start_id = int(sys.argv[1]);
step = int(sys.argv[2]);

slide_list_file = '/data08/shared/lehhou/necrosis_segmentation_workingdir/slide_list.txt';
#slide_list_file = './test_data/temp_slide_list.txt';

def load_model_value(model_file):
    loaded_var = pickle.load(open(model_file, 'rb'));
    #global mu;
    #global sigma;
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

def Segment_Necrosis_WholeSlide():
    print "start_id, step: ", start_id, step;
    # Load model file (should be put in upper level later)
    print ('Load model file...');
    mu, sigma, param_values = load_model_value(classification_model_file);
    print ('Finish loading model file');

    with open(slide_list_file) as f:
        slide_list = f.readlines();
    slide_list = [x.strip() for x in slide_list];

    for slide_path in slide_list[start_id::step]:
        # Check existing heatmap
        if (slide_path.endswith('/')):
            slide_path = slide_path[:-1];

        parent_path, slide_name = os.path.split(slide_path);
        heatmap_folder = slide_path;
        heatmap_path = heatmap_folder + '/patch-level-necrosis.txt';
        #if (not os.path.isfile(heatmap_path)):
        #print ("{0}\n{1}\n{2}\n\n".format(slide_path, slide_name, heatmap_path));


        if (not os.path.isfile(heatmap_path)):
            print "Processing ", slide_path;
            #print ("{}\n{}\n{}\n\n".format(slide_path, slide_name, heatmap_path;
            predict_slide(slide_path, mu, sigma, param_values, heatmap_folder);


if __name__ == "__main__":
    Segment_Necrosis_WholeSlide();
