import numpy as np
import theano
import theano.tensor as T

#a = T.tensor3('a');
#b = T.matrix('b');
#f = theano.function([a, b], T.sum(a * b, axis=2));
#print f([[[1, 2], \
#          [3, 4]], \
#          \
#         [[5, 6], \
#          [7, 8]]], \
#          \
#         [[1, 2], \
#          [3, 4]]);

#a = T.matrix('a');
#b = T.matrix('b');
#f = theano.function([a, b], a * b[1, :]);
#print f([[1, 2], \
#         [3, 4], \
#         [5, 6], \
#         [7, 8]], \
#          \
#        [[1, 2], \
#         [2, 3], \
#         [4, 5]]);

#a = T.matrix('a');
#x_start = 3;
#x_end = 4.5;
#left = T.le(x_start, a);
#right = T.gt(x_end, a);
#c = 0 * (1 - left) + (a - x_start) * left * right + (x_end - x_start) * (1 - right);
#f = theano.function([a], c);
#print f([[1, 2], [3, 4], [5, 6]]);

#a = T.matrix('a');
#b = T.vector('b');
#le = T.le(a, b);
#f = theano.function([a, b], le);
#print f([[1, 2], [1, 4], [5, 1]], [2, 3]);

#a = T.tensor3('a');
#b = a.T;
#f = theano.function([a], b);
#x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32);
#print x;
#print f(x);

#size = 4;
#x = T.tensor4('x');
#ind = T.as_tensor_variable(np.indices((size, size))) - (size - 1.0) / 2.0;
#theta = 0.25 * np.pi;
#rotate = T.stack(T.cos(theta), -T.sin(theta), T.sin(theta), T.cos(theta)).reshape((2, 2));
#ind_rot = T.tensordot(rotate, ind, axes=((0, 0))) + (size - 1.0) / 2.0;
#transy = T.clip(ind_rot[0], 0, size - 1 - .001);
#transx = T.clip(ind_rot[1], 0, size - 1 - .001);
#vert = T.iround(transy);
#horz = T.iround(transx);
#y = x[:, :, vert, horz];
#
#f = theano.function([x], [y, ind_rot]);
#a = np.random.random((2, 1, size, size)).astype(np.float32);
#b, r = f(a);
#
#print a;
#print '------------------';
#print b;
#print '------------------';
#print np.indices((size, size));
#print '------------------';
#print np.round(r).astype(np.int);

#mytype = T.TensorType('float32', [True, False, True, True])
#a = mytype('a');
#b = T.tensor4('b');
#c = b * a;
#f = theano.function([a, b], [c]);
#
#x = np.random.random((1, 2, 1, 1)).astype(np.float32);
#y = np.random.random((2, 2, 3, 2)).astype(np.float32);
#print x;
#print '--------------------';
#print y;
#print '--------------------';
#print f(x, y);

#a = T.matrix('a');
#b = T.matrix('b');
#c = b * T.shape_padleft(a, n_ones=0);
#f = theano.function([a, b], [c]);
#
#x = np.random.random((2, 2)).astype(np.float32);
#y = np.random.random((2, 2)).astype(np.float32);
#print x;
#print '--------------------';
#print y;
#print '--------------------';
#print f(x, y);

a = T.tensor3('a');
b = T.sum(a, axis = 1, keepdims = True);
f = theano.function([a], [b]);
a = np.random.random((3, 2, 3)).astype(np.float32);
print a;
print f(a);

