import theano.tensor as T
import numpy as np
from lasagne import init, nonlinearities
from lasagne.utils import as_tuple
from lasagne.theano_extensions import conv, padding
from lasagne.layers import Layer


def conv_output_length(input_length, filter_size,
                       stride, border_mode, pad=0):
    """
    Helper function to compute the output size of a convolution operation
    """
    if input_length is None:
        return None
    if border_mode == 'valid':
        output_length = input_length - filter_size + 1
    elif border_mode == 'full':
        output_length = input_length + filter_size - 1
    elif border_mode == 'same':
        output_length = input_length
    elif border_mode == 'pad':
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid border mode: {0}'.format(border_mode))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


class RotConv(Layer):
    """
    Rotation invariant 2D convolutional layer
    """
    def __init__(self, incoming, num_filters, num_rot,
                 filter_size, stride=(1, 1),
                 border_mode="valid", untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(RotConv, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.num_rot = num_rot;
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.border_mode = border_mode
        self.untie_biases = untie_biases
        self.convolution = convolution

        if self.border_mode not in ['valid', 'full', 'same']:
            raise RuntimeError("Invalid border mode: '%s'" % self.border_mode)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def rot_filters(self, theta):
        fsize = self.filter_size[0];
        ind = T.as_tensor_variable(np.indices((fsize, fsize)) - (fsize - 1.0) / 2.0);
        rotate = T.stack(T.cos(theta), -T.sin(theta), T.sin(theta), T.cos(theta)).reshape((2, 2));
        ind_rot = T.tensordot(rotate, ind, axes=((0, 0))) + (fsize - 1.0) / 2.0;
        transy = T.clip(ind_rot[0], 0, fsize - 1 - .00001);
        transx = T.clip(ind_rot[1], 0, fsize - 1 - .00001);
        vert = T.iround(transy);
        horz = T.iround(transx);
        return self.W[:, :, vert, horz];

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         self.border_mode)

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            self.border_mode)

        return (input_shape[0], self.num_filters * self.num_rot, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        filter_shape = self.get_W_shape()

        if self.border_mode in ['valid', 'full']:
            conv_rot_list = [];
            for r in range(self.num_rot):
                conv_rot_list.append(self.convolution(input, self.rot_filters(2 * np.pi * r / self.num_rot),
                                                      subsample=self.stride,
                                                      image_shape=input_shape,
                                                      filter_shape=filter_shape,
                                                      border_mode=self.border_mode));

            conved = T.concatenate(conv_rot_list, axis=1);
        elif self.border_mode == 'same':
            if self.stride != (1, 1):
                raise NotImplementedError("Strided convolution with "
                                          "border_mode 'same' is not "
                                          "supported by this layer yet.")

            conv_rot_list = [];
            for r in range(self.num_rot):
                conv_rot_list.append(self.convolution(input, self.rot_filters(2 * np.pi * r / self.num_rot),
                                                      subsample=self.stride,
                                                      image_shape=input_shape,
                                                      filter_shape=filter_shape,
                                                      border_mode='full'));

            conved = T.concatenate(conv_rot_list, axis=1);

            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            b_list = [];
            for r in range(self.num_rot):
                b_list.append(self.b.dimshuffle('x', 0, 1, 2));
            b_cat = T.concatenate(b_list, axis=1);
            activation = conved + b_cat;
        else:
            b_list = [];
            for r in range(self.num_rot):
                b_list.append(self.b.dimshuffle('x', 0, 'x', 'x'));
            b_cat = T.concatenate(b_list, axis=1);
            activation = conved + b_cat;

        return self.nonlinearity(activation)


