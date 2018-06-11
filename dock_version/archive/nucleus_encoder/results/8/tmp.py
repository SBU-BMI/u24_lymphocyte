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

a = T.matrix('a');
x_start = 3;
x_end = 4.5;
left = T.le(x_start, a);
right = T.gt(x_end, a);
c = 0 * (1 - left) + (a - x_start) * left * right + (x_end - x_start) * (1 - right);
f = theano.function([a], c);
print f([[1, 2], [3, 4], [5, 6]]);

