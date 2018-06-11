import lasagne
import theano.tensor as T
import numpy as np

# The input must be flattened
class SmthActLayer(lasagne.layers.Layer):
    def __init__(self, incoming, x_start, x_end, num_segs, W = lasagne.init.Normal(0.01), **kwargs):
        super(DotLayer, self).__init__(incoming, **kwargs);
        num_inputs = self.input_shape[1];

        self.x_start = x_start;
        self.x_end = x_end;
        self.x_step = (x_end - x_start) / num_segs;
        self.num_segs = num_segs;

        self.W = self.add_param(W, (num_segs, num_inputs), name = 'W', small_weights = True);

    def basisf(x, start, end):
        ab_start = T.le(x_start, x);
        lt_end = T.gt(x_end, x);
        return 0 * (1 - ab_start) + (x - x_start) * ab_start * lt_end + (x_end - x_start) * (1 - lt_end);

    def get_output_for(self, input, **kwargs):
        output = T.zeros_like(input);
        for s in range(self.num_segs):
            output += basisf(output, self.x_start + self.x_step * s, self.x_start + self.x_step * (s + 1)) * self.W[s, :];

        return output;

