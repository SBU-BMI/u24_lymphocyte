from nolearn.lasagne import BatchIterator
import numpy as np
from scipy import misc
from PIL import Image
from skimage.color import hed2rgb, rgb2hed

APS = 100;
PS = 60;
MARGIN = 15;

def rotate_img(x, degree):
    rotate_angle = (np.random.rand(1)[0]-0.5)*2 * degree;
    im = Image.fromarray(x.astype(np.uint8));
    x = np.array(im.rotate(rotate_angle, Image.BICUBIC)).astype(np.float32);
    return x;

def data_aug_img(img, mu, sigma, deterministic=False, idraw=-1, jdraw=-1):
    # rotate small degree
    if np.random.rand(1)[0] < 0.9:
        img = img.transpose();
        if deterministic:
            img = rotate_img(img, 10.0);
        else:
            img = rotate_img(img, 45.0);
        img = img.transpose();

    # crop
    icut = APS - PS;
    jcut = APS - PS;
    if deterministic:
        if idraw < -0.5 or jdraw < -0.5:
            ioff = int(icut // 2);
            joff = int(jcut // 2);
        else:
            ioff = idraw;
            joff = jdraw;
    else:
        ioff = np.random.randint(MARGIN, icut + 1 - MARGIN);
        joff = np.random.randint(MARGIN, jcut + 1 - MARGIN);
    img = img[:, ioff : ioff+PS, joff : joff+PS];

    # adjust color
    if not deterministic:
        adj_add = np.array([[[0.15, 0.15, 0.02]]], dtype=np.float32);
        img = np.clip(hed2rgb( \
                rgb2hed(img.transpose((2, 1, 0)) / 255.0) + np.random.uniform(-1.0, 1.0, (1, 1, 3))*adj_add \
              ).transpose((2, 1, 0))*255.0, 0.0, 255.0);

    if not deterministic:
        adj_range = 0.1;
        adj_add = 5;
        rgb_mean = np.mean(img, axis=(1,2), keepdims=True).astype(np.float32);
        adj_magn = np.random.uniform(1 - adj_range, 1 + adj_range, (3, 1, 1)).astype(np.float32);
        img = np.clip((img-rgb_mean)*adj_magn + rgb_mean + np.random.uniform(-1.0, 1.0, (3, 1, 1))*adj_add, 0.0, 255.0);

    # mirror and flip
    if np.random.rand(1)[0] < 0.5:
        img = img[:, ::-1, :];
    if np.random.rand(1)[0] < 0.5:
        img = img[:, :, ::-1];

    # transpose
    if np.random.rand(1)[0] < 0.5:
        img = img.transpose((0, 2, 1));

    ## scaling
    #if not deterministic:
    #    if np.random.rand(1)[0] < 0.1:
    #        iscale = 2*(np.random.rand(1)[0]-0.5)*0.05 + 1.0;
    #        jscale = 2*(np.random.rand(1)[0]-0.5)*0.05 + 1.0;
    #        img = misc.imresize(img.transpose().astype(np.uint8), \
    #                (int(img.shape[2]*jscale), int(img.shape[1]*iscale)) \
    #                ).transpose().astype(np.float32);

    img = zero_centering(img);
    img = (img / 255.0 - mu) / sigma;

    return img;

def zero_centering(img):
    x0 = (img.shape[1] - PS) // 2;
    y0 = (img.shape[2] - PS) // 2;
    im = Image.fromarray(img.transpose().astype(np.uint8));
    img = np.array(im.crop((x0, y0, x0+PS, y0+PS))).transpose().astype(np.float32);
    return img;

def data_aug(X, mu, sigma, deterministic=False, idraw=-1, jdraw=-1):
    Xc = np.zeros(shape=(X.shape[0], X.shape[1], PS, PS), dtype=np.float32);
    Xcopy = X.copy();
    for i in range(len(Xcopy)):
        Xc[i] = data_aug_img(Xcopy[i], mu, sigma, deterministic=deterministic, idraw=idraw, jdraw=jdraw);
    return Xc;

