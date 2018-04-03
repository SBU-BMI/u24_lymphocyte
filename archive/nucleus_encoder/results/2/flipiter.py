from nolearn.lasagne import BatchIterator
import numpy as np

class FlipBatchIterator(BatchIterator):
    def transform(self, X1, X2):
        X1b, X2b = super(FlipBatchIterator, self).transform(X1, X2);
        X2b = X2b.reshape(X1b.shape);

        bs = X1b.shape[0];
        h_indices = np.random.choice(bs, bs / 2, replace=False);  # horizontal flip
        v_indices = np.random.choice(bs, bs / 2, replace=False);  # vertical flip
        r_indices = np.random.choice(bs, bs / 2, replace=False);  # 90 degree rotation

        for X in (X1b, X2b):
            X[h_indices] = X[h_indices, :, :, ::-1];
            X[v_indices] = X[v_indices, :, ::-1, :];
            X[r_indices] = np.swapaxes(X[r_indices, :, :, :], 2, 3);
        shape = X2b.shape;
        X2b = X2b.reshape((shape[0], -1));

        return X1b, X2b;

