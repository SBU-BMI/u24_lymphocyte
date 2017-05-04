import lasagne
import theano.tensor as T
import numpy as np

# The input must be flattened
class SmthActLayer(lasagne.layers.Layer):
    def __init__(self, incoming, x_start, x_end, num_segs, W = lasagne.init.Normal(0.01), **kwargs):
        super(SmthActLayer, self).__init__(incoming, **kwargs);
        num_inputs = self.input_shape[1];

        self.x_start = x_start;
        self.x_end = x_end;
        self.x_step = (x_end - x_start) / num_segs;
        self.num_segs = num_segs;

        self.W = self.add_param(W, (num_segs, num_inputs), name = 'W', small_weights = True);

    def basisf(self, x, start, end):
        ab_start = T.le(start, x);
        lt_end = T.gt(end, x);
        return 0 * (1 - ab_start) + (x - start) * ab_start * lt_end + (end - start) * (1 - lt_end);

    def get_output_for(self, input, **kwargs):
        output = T.zeros_like(input);
        for seg in range(0, self.num_segs):
            output += self.basisf(input, self.x_start + self.x_step * seg, self.x_start + self.x_step * (seg + 1)) * self.W[seg, :];

        return output;


class DotLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
        super(DotLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, 2 * num_units), name='W')

    def get_output_for(self, input, **kwargs):
        return T.dot(input, self.W[:, :self.num_units]) + T.dot(input, self.W[:, self.num_units:]);

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

