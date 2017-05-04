import numpy as np
import lasagne
import pickle
import io
import matplotlib.pyplot as plt
import urllib

from scipy.misc import imresize
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import rectify, sigmoid, identity, very_leaky_rectify, softmax, leaky_rectify
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX
from scipy import misc

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    output_layer = net['prob'];

    return output_layer

output_layer = build_model();
model = pickle.load(open('vgg16.pkl'))

lasagne.layers.set_all_param_values(output_layer, model['param values']);

# Read an image from web
url = 'https://i.ytimg.com/vi/KY4IzMcjX3Y/maxresdefault.jpg';
ext = url.split('.')[-1]
im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)

# Resize and crop the image
h, w, _ = im.shape
size_ratio = max(256.0 / h, 256.0 / w);
im = imresize(im, size_ratio);
h, w, _ = im.shape
im = im[h//2-112:h//2+112, w//2-112:w//2+112];

# Prepare the format
im = im.transpose()[::-1, :, :].astype(np.float32);
im[0, :, :] = im[0, :, :] - model['mean value'][0];
im[1, :, :] = im[1, :, :] - model['mean value'][1];
im[2, :, :] = im[2, :, :] - model['mean value'][2];

# Put the input into VGG, and get top5 predictions
prob = np.array(lasagne.layers.get_output(output_layer, im[np.newaxis], deterministic=True).eval());
top5 = np.argsort(prob[0])[-1:-6:-1];

# Print the meanings of top5 predictions
for n, label in enumerate(top5):
    print n+1, model['synset words'][label];

