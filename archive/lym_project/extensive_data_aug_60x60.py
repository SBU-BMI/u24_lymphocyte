from nolearn.lasagne import BatchIterator
import numpy as np
from scipy import misc
from PIL import Image

PS = 60;

def data_aug_img(img, deterministic=False):
    # adjust color
    if not deterministic:
        adj_range = 0.05;
        adj_add = 0.01;
        rgb_mean = np.mean(img, axis=(1,2), keepdims=True).astype(np.float32);
        adj_magn = np.random.uniform(1-adj_range, 1+adj_range, (3, 1, 1)).astype(np.float32);
        img = (img-rgb_mean) * adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (3, 1, 1)) * adj_add;

    # mirror and flip
    if not deterministic:
        if np.random.rand(1)[0] < 0.5:
            img = img[:, ::-1, :];
        if np.random.rand(1)[0] < 0.5:
            img = img[:, :, ::-1];

    # transpose
    if not deterministic:
        if np.random.rand(1)[0] < 0.5:
            img = img.transpose((0, 2, 1));

    return img;

def data_aug(X, deterministic=False):
    Xc = np.zeros(shape=(X.shape[0], X.shape[1], PS, PS), dtype=np.float32);
    Xcopy = X.copy();
    for i in range(len(Xcopy)):
        Xc[i] = data_aug_img(Xcopy[i], deterministic=deterministic);
    return Xc;

