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


def exc_test(test_func, X_test, a_test, y_test, APS, PS):
    print("Start testing...");

    #i_draw_arr = [0,150,300];
    #j_draw_arr = [0,150,300];
    i_draw_arr = [0, (APS - PS)/2, APS - PS];
    j_draw_arr = [0, (APS - PS)/2, APS - PS];
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
        input_real_org, aug_real_org, target_real_org = batch;
        #input_real = data_aug(input_real, mu, sigma);

        image_patch = np.zeros(shape=(input_real_org.shape[0], 3, APS, APS), dtype=np.float32);
        prediction_patch = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);
        groundtruth_patch = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);

        weight_2d = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);
        mask_2d = np.ones(shape=(input_real_org.shape[0], PS, PS), dtype=np.float32);
        weight_3d = np.zeros(shape=(input_real_org.shape[0], 3, APS, APS), dtype=np.float32);
        mask_3d = np.ones(shape=(input_real_org.shape[0], 3, PS, PS), dtype=np.float32);

        print "image_patch, mask_3d ", image_patch.shape, mask_3d.shape;

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
                print prediction_real.shape;
                prediction_patch[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += np.reshape(prediction_real, (-1, PS, PS));
                groundtruth_patch[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += np.reshape(target_real, (-1, PS, PS));
                weight_2d[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += mask_2d;
                weight_3d[:, :, i_draw:i_draw+PS, j_draw:j_draw+PS] += mask_3d;

                total_error += error;
                i_batch += 1;
                i_line += len(prediction_real);

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

