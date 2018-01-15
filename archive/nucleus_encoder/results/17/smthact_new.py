import lasagne
import theano.tensor as T
import numpy as np

class SmthActLayer(lasagne.layers.Layer):
    def __init__(self, incoming, x_start, x_end, num_segs, order = 2, tied_feamap = True,
                 W = lasagne.init.Normal(std = 0.2, mean = 0.0), **kwargs):
        super(SmthActLayer, self).__init__(incoming, **kwargs);
        input_dim = self.input_shape;
        self.x_start = x_start;
        self.x_end = x_end;
        self.x_step = float(x_end - x_start) / num_segs;
        self.num_segs = num_segs;
        self.order = order;
        self.tied_feamap = tied_feamap;

        if self.tied_feamap:
            W_dim = (num_segs, input_dim[1]);
        else:
            W_dim = (num_segs,) + input_dim[1::];
        self.W = self.add_param(W, W_dim, name = 'W', small_weights = True, regularizable = True);

    def basisf(self, x, s, e):
        cpstart = T.le(s, x);
        cpend = T.gt(e, x);
        if self.order == 1:
            return 0 * (1 - cpstart) + (x - s) * cpstart * cpend + (e - s) * (1 - cpend);
        else:
            return 0 * (1 - cpstart) + 0.5 * (x - s)**2 * cpstart * cpend + ((e - s) * (x - e) + 0.5 * (e - s)**2) * (1 - cpend);

    def get_output_for(self, input, **kwargs):
        output = T.zeros_like(input);
        for seg in range(0, self.num_segs):
            if self.tied_feamap:
                output += self.basisf(input, self.x_start + self.x_step * seg, self.x_start + self.x_step * (seg + 1)) \
                        * T.shape_padleft(T.shape_padright(self.W[seg], n_ones = len(input_dim) - 2));
            else:
                output += self.basisf(input, self.x_start + self.x_step * seg, self.x_start + self.x_step * (seg + 1)) \
                        * T.shape_padleft(self.W[seg]);

        return output;


class AgoLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_segs, tied_feamap = True,
                 W = lasagne.init.Normal(std = 0.2, mean = 0.0),
                 P = lasagne.init.Normal(std = 5, mean = 0.0), **kwargs):
        super(AgoLayer, self).__init__(incoming, **kwargs);
        input_dim = self.input_shape;
        self.num_segs = num_segs;
        self.tied_feamap = tied_feamap;

        if self.tied_feamap:
            W_dim = (num_segs, input_dim[1]);
        else:
            W_dim = (num_segs,) + input_dim[1::];
        self.W = self.add_param(W, W_dim, name = 'W', regularizable = True);
        self.P = self.add_param(P, W_dim, name = 'P', regularizable = True);

    def basisf(self, x, bks):
        act_bks = T.gt(x, bks);
        return x * act_bks;

    def get_output_for(self, input, **kwargs):
        output = input * T.gt(input, 0);
        for seg in range(0, self.num_segs):
            if self.tied_feamap:
                output += self.basisf(input, T.shape_padleft(T.shape_padright(self.P[seg], n_ones = len(input_dim) - 2))) \
                        * T.shape_padleft(T.shape_padright(self.W[seg], n_ones = len(input_dim) - 2));
            else:
                output += self.basisf(input, T.shape_padleft(self.P[seg])) \
                        * T.shape_padleft(self.W[seg]);

        return output;


class HeLayer(lasagne.layers.Layer):
    def __init__(self, incoming, tied_feamap = True,
                 W = lasagne.init.Normal(std = 0.01, mean = -0.2), **kwargs):
        super(HeLayer, self).__init__(incoming, **kwargs);
        input_dim = self.input_shape;
        self.tied_feamap = tied_feamap;

        if self.tied_feamap:
            W_dim = (input_dim[1],);
        else:
            W_dim = input_dim[1::];
        self.W = self.add_param(W, W_dim, name = 'W', regularizable = True);

    def get_output_for(self, input, **kwargs):
        if self.tied_feamap:
            return input * T.gt(input, 0) + input * T.le(input, 0) \
                 * T.shape_padleft(T.shape_padright(self.W[seg], n_ones = len(input_dim) - 2));
        else:
            return input * T.gt(input, 0) + input * T.le(input, 0) \
                 * T.shape_padleft(self.W);

