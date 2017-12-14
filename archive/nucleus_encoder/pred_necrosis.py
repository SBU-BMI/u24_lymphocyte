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
from pred_necrosis_wsi import predict_slide

import scipy.misc
import os.path

model_idx = 270;
classification_model_file = 'model_vals/deep_conv_classification_model_deep_segmentation_deconv_necrosis_alt2_e{}.pkl'.format(model_idx);
slide_path = sys.argv[1];

def load_model_value(model_file):
    loaded_var = pickle.load(open(model_file, 'rb'));
    mu = loaded_var[0];
    sigma = loaded_var[1];

    param_values = loaded_var[2];
    '''
    param_set = lasagne.layers.get_all_params(network);
    for param, value in zip(param_set, param_values):
        param.set_value(value);
    '''
    return mu, sigma, param_values;


def Segment_Necrosis_WholeSlide():
    mu, sigma, param_values = load_model_value(classification_model_file);
    predict_slide(slide_path, mu, sigma, param_values, heatmap_folder);


if __name__ == "__main__":
    Segment_Necrosis_WholeSlide();


