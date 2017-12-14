import numpy as np
from scipy import misc
from PIL import Image
from skimage.color import hed2rgb, rgb2hed

APS = 100;
PS = 100;
MARGIN = 90;

def rotate_img(x, degree):
    rotate_angle = (np.random.rand(1)[0]-0.5)*2 * degree;
    im = Image.fromarray(x.astype(np.uint8));
    x = np.array(im.rotate(rotate_angle, Image.BICUBIC)).astype(np.float32);
    return x;

def data_aug_img(img, mu, sigma, deterministic=False, idraw=-1, jdraw=-1):
    # mirror and flip
    if np.random.rand(1)[0] < 0.5:
        img = img[:, ::-1, :];
    if np.random.rand(1)[0] < 0.5:
        img = img[:, :, ::-1];

    # transpose
    if np.random.rand(1)[0] < 0.5:
        img = img.transpose((0, 2, 1));

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

