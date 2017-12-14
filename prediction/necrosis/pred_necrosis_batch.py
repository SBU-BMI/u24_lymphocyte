import numpy as np
import lasagne
import theano

from lasagne import layers
from lasagne.nonlinearities import sigmoid, rectify, leaky_rectify, identity
from data_aug_necrosis import data_aug
from batch_norms import batch_norm
from shape import ReshapeLayer


# Parameters
APS = None;
PS = None;
batchsize = 10;
LearningRate = theano.shared(np.array(1e-3, dtype=np.float32));

mu = None;
sigma = None;

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


def build_function(network, param_set, input_var, target_var):
    prediction_var_train = lasagne.layers.get_output(network, deterministic=False);
    prediction_var_test = lasagne.layers.get_output(network, deterministic=True);
    loss_train = lasagne.objectives.squared_error(prediction_var_train, target_var).mean();
    loss_test = lasagne.objectives.squared_error(prediction_var_test, target_var).mean();

    # training function
    updates = lasagne.updates.nesterov_momentum(loss_train, param_set, learning_rate=LearningRate, momentum=0.985);
    train_func = theano.function([input_var, target_var], loss_train, updates=updates);

    test_func = theano.function([input_var, target_var], [loss_test, prediction_var_test]);

    print("finish building function");
    return train_func, test_func;

def test_all(X_test, y_test, loaded_mu, loaded_sigma, train_func, test_func):
    global mu;
    global sigma;
    mu = loaded_mu;
    sigma = loaded_sigma;

    # Generating augmenting data
    a_test = np.zeros(shape=(X_test.shape[0], 1), dtype=np.float32);

    # do testing
    image_array, groundtruth_array, prediction_array = exc_test(test_func, X_test, a_test, y_test);

    return image_array, groundtruth_array, prediction_array;

def exc_test(test_func, X_test, a_test, y_test):
    print("Start testing...");

    i_draw_arr = [0, APS - PS];
    j_draw_arr = [0, APS - PS];

    total_error = 0;
    i_batch = 0;
    i_line = 0;

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

        image_patch = np.zeros(shape=(input_real_org.shape[0], 3, APS, APS), dtype=np.float32);
        prediction_patch = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);
        groundtruth_patch = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);

        weight_2d = np.zeros(shape=(input_real_org.shape[0], APS, APS), dtype=np.float32);
        mask_2d = np.ones(shape=(input_real_org.shape[0], PS, PS), dtype=np.float32);
        weight_3d = np.zeros(shape=(input_real_org.shape[0], 3, APS, APS), dtype=np.float32);
        mask_3d = np.ones(shape=(input_real_org.shape[0], 3, PS, PS), dtype=np.float32);

        temp_idx = 0;
        for i_draw in i_draw_arr:
            for j_draw in j_draw_arr:
                input_real, target_real = data_aug(X=input_real_org, Y=target_real_org, \
                        mu=mu, sigma=sigma, deterministic=True, \
                        idraw=i_draw, jdraw=j_draw, APS=APS, PS=PS);

                target_real = target_real.reshape(target_real.shape[0], -1);
                error, prediction_real = test_func(input_real, target_real);

                class_res = from_pred_to_class(prediction_real);
                class_res_flattened = class_res.reshape(-1, 1);

                class_array[i_line:i_line+len(prediction_real)] = class_res;
                image_patch[:, :,i_draw:i_draw+PS, j_draw:j_draw+PS] += input_real;

                prediction_patch[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += np.reshape(prediction_real, (-1, PS, PS));
                groundtruth_patch[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += np.reshape(target_real, (-1, PS, PS));
                weight_2d[:, i_draw:i_draw+PS, j_draw:j_draw+PS] += mask_2d;
                weight_3d[:, :, i_draw:i_draw+PS, j_draw:j_draw+PS] += mask_3d;

                total_error += error;
                i_batch += 1;
                i_line += len(prediction_real);

                temp_idx += 1;

        image_patch = np.divide(image_patch, weight_3d);
        prediction_patch = np.divide(prediction_patch, weight_2d);
        groundtruth_patch = np.divide(groundtruth_patch, weight_2d);

        image_array[begin_img_idx:begin_img_idx+input_real_org.shape[0]] = image_patch;
        prediction_array[begin_img_idx:begin_img_idx+input_real_org.shape[0]] = prediction_patch;
        groundtruth_array[begin_img_idx:begin_img_idx+input_real_org.shape[0]] = groundtruth_patch;

        begin_img_idx += input_real_org.shape[0];

    return image_array, groundtruth_array, prediction_array;


def from_pred_to_class(pred):
    class_res = np.copy(pred);
    class_res = (class_res >= 0.5).astype(np.int32);
    return class_res;


def necrosis_predict(X_test, Y_test, loaded_mu, loaded_sigma, param_values, loaded_APS, loaded_PS):
    # attach additional layers for classification
    network, input_var, output_var, all_param = build_deconv_network_temp();

    # load param values of the encoder
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
    train_func, test_func = build_function(network, all_param, input_var, target_var);

    # run training  (X_test, y_test, loaed_mu, loaded_sigma, train_func, test_func)
    image_array, groundtruth_array, prediction_array = test_all(X_test, Y_test, mu, sigma, train_func, test_func);

    print "DONE!";

    return image_array, groundtruth_array, prediction_array;

