import numpy as np
from PIL import Image
from skimage.color import hed2rgb, rgb2hed

MARGIN = 10;

def data_aug_img(img, msk, mu, sigma, deterministic=False, idraw=-1, jdraw=-1, APS=500, PS=200):
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
    msk = msk[ioff : ioff+PS, joff : joff+PS];

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
    if not deterministic:
        if np.random.rand(1)[0] < 0.5:
            img = img[:, ::-1, :];
            msk = msk[::-1, :];
        if np.random.rand(1)[0] < 0.5:
            img = img[:, :, ::-1];
            msk = msk[:, ::-1];

    # transpose
    if not deterministic:
        if np.random.rand(1)[0] < 0.5:
            img = img.transpose((0, 2, 1));
            msk = msk.transpose((1, 0));

    img, msk = zero_centering(img, msk, APS=APS, PS=PS);
    img = (img - mu) / sigma;

    return img, msk;

def zero_centering(img, msk, APS=500, PS=224):
    x0 = (img.shape[1] - PS) // 2;
    y0 = (img.shape[2] - PS) // 2;
    im = Image.fromarray(img.transpose().astype(np.uint8));
    mk = Image.fromarray(msk.transpose().astype(np.uint8));
    img = np.array(im.crop((x0, y0, x0+PS, y0+PS))).transpose().astype(np.float32);
    msk = np.array(mk.crop((x0, y0, x0+PS, y0+PS))).transpose().astype(np.float32);
    return img, msk;

def data_aug(X, Y, mu, sigma, deterministic=False, idraw=-1, jdraw=-1, APS=500, PS=200):
    Xc = np.zeros(shape=(X.shape[0], X.shape[1], PS, PS), dtype=np.float32);
    Yc = np.zeros(shape=(Y.shape[0], PS, PS), dtype=np.int32);
    Xcopy = X.copy();
    Ycopy = Y.copy();
    for i in range(len(Xcopy)):
        Xc[i], Yc[i] = data_aug_img(Xcopy[i], Ycopy[i], mu, sigma, deterministic=deterministic, idraw=idraw, jdraw=jdraw, APS=APS, PS=PS);
    return Xc, Yc;

