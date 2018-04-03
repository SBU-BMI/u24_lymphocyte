import lasagne
import theano.tensor as T
import numpy as np

class ChInnerProd(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ChInnerProd, self).__init__(incoming, **kwargs);

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape);
        output_shape[1] -= 1;
        return tuple(output_shape);

    def get_output_for(self, input, **kwargs):
        output = T.shape_padaxis(input[:, 0], axis=1) * input[:, 1:];
        return output;

class ChInnerProdMerge(lasagne.layers.MergeLayer):
    def __init__(self, feat_map, mask_map, **kwargs):
        super(ChInnerProdMerge, self).__init__([feat_map, mask_map], **kwargs);

    def get_output_shape_for(self, input_shapes):
        output_shape = list(input_shapes[0]);
        return tuple(output_shape);

    def get_output_for(self, inputs, **kwargs):
        output = inputs[0] * inputs[1];
        return output;

class CenterCrop(lasagne.layers.Layer):
    def __init__(self, incoming, crop_size, **kwargs):
        super(CenterCrop, self).__init__(incoming, **kwargs);
        self.crop_size = crop_size;

    def get_output_for(self, input, **kwargs):
        return input[:, :, self.crop_size:-self.crop_size, self.crop_size:-self.crop_size];

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape);
        return (input_shape[0], input_shape[1], input_shape[2]-2*self.crop_size, input_shape[3]-2*self.crop_size);

