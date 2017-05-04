import lasagne
import theano.tensor as T
import numpy as np

# The input must be flattened
class SmthAct1Layer(lasagne.layers.Layer):
    def __init__(self, incoming, x_start, x_end, num_segs, W = lasagne.init.Normal(std = 0.2, mean = 0.0), **kwargs):
        super(SmthAct1Layer, self).__init__(incoming, **kwargs);
        num_inputs = self.input_shape[1];

        self.x_start = x_start;
        self.x_end = x_end;
        self.x_step = float(x_end - x_start) / num_segs;
        self.num_segs = num_segs;

        self.W = self.add_param(W, (num_segs, num_inputs), name = 'W', small_weights = True, regularizable = True);

    def basisf(self, x, s, e):
        cpstart = T.le(s, x);
        cpend = T.gt(e, x);
        return 0 * (1 - cpstart) + (x - s) * cpstart * cpend + (e - s) * (1 - cpend);

    def get_output_for(self, input, **kwargs):
        output = T.zeros_like(input);
        for seg in range(0, self.num_segs):
            output += self.basisf(input, self.x_start + self.x_step * seg, self.x_start + self.x_step * (seg + 1)) * self.W[seg, :];

        return output;


class SmthAct2Layer(lasagne.layers.Layer):
    def __init__(self, incoming, x_start, x_end, num_segs, W = lasagne.init.Normal(std = 0.01, mean = 0.1), **kwargs):
        super(SmthAct2Layer, self).__init__(incoming, **kwargs);
        num_inputs = self.input_shape[1];

        self.x_start = x_start;
        self.x_end = x_end;
        self.x_step = float(x_end - x_start) / num_segs;
        self.num_segs = num_segs;

        self.W = self.add_param(W, (num_segs, num_inputs), name = 'W', small_weights = True, regularizable = True);

    def basisf(self, x, s, e):
        cpstart = T.le(s, x);
        cpend = T.gt(e, x);
        return 0 * (1 - cpstart) + 0.5 * (x - s)**2 * cpstart * cpend + ((e - s) * (x - e) + 0.5 * (e - s)**2) * (1 - cpend);

    def get_output_for(self, input, **kwargs):
        output = T.zeros_like(input);
        for seg in range(0, self.num_segs):
            output += self.basisf(input, self.x_start + self.x_step * seg, self.x_start + self.x_step * (seg + 1)) * self.W[seg, :];

        return output;


class AgoLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_segs, W = lasagne.init.Normal(std = 0.2, mean = 0.0),
                 P = lasagne.init.Normal(std = 5, mean = 0.0), **kwargs):
        super(AgoLayer, self).__init__(incoming, **kwargs);
        num_inputs = self.input_shape[1];

        self.num_segs = num_segs;
        self.W = self.add_param(W, (num_segs, num_inputs), name = 'W', regularizable = True);
        self.P = self.add_param(P, (num_segs, num_inputs), name = 'P', regularizable = True);

    def basisf(self, x, bks):
        act_bks = T.gt(x, bks);
        return x * act_bks;

    def get_output_for(self, input, **kwargs):
        output = input * T.gt(input, 0);
        for seg in range(0, self.num_segs):
            output += self.basisf(input, self.P[seg, :]) * self.W[seg, :];

        return output;


class HeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, W = lasagne.init.Normal(std = 0.01, mean = -0.2), **kwargs):
        super(HeLayer, self).__init__(incoming, **kwargs);
        num_inputs = self.input_shape[1];
        self.W = self.add_param(W, (num_inputs,), name = 'W', regularizable = True);

    def get_output_for(self, input, **kwargs):
        return input * T.gt(input, 0) + input * T.le(input, 0) * T.shape_padleft(self.W, n_ones=1);

