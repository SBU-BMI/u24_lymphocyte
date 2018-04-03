import numpy as np
from numpy.random import uniform, chisquare, gamma
from scipy.misc import toimage
import math

def add_ellipse(mask, cx, cy, theta, a, b):
    im_size = mask.shape[0];
    for i in range(im_size):
        for j in range(im_size):
            x = i - cx;
            y = j - cy;
            xp = x * math.cos(theta) + y * math.sin(theta);
            yp = -x * math.sin(theta) + y * math.cos(theta);
            if (xp / a)**2 + (yp / b)**2 <= 1:
                mask[i, j] = 1;
    return mask;

def rand_mask(im_size):
    mask = np.zeros(shape=(im_size, im_size), dtype=np.uint8);
    cx = (im_size - 1) / 2.0;
    cy = (im_size - 1) / 2.0;
    a = im_size * (gamma(2.2, 0.6) / 10.0 + 0.04);
    b = im_size * (gamma(2.2, 0.6) / 10.0 + 0.04);
    ratio = float(max(a, b)) / min(a, b);
    mask = add_ellipse(mask, cx, cy, uniform() * math.pi, a, b);
    for i in range(np.random.randint(2, 5)):
        x = cx;
        y = cy;
        while ((x - cx)**2 + (y - cy)**2)**0.5 < im_size * 0.3:
            x = np.random.randint(0, im_size);
            y = np.random.randint(0, im_size);

        a = im_size * (gamma(2.2, 0.6) / 10.0 + 0.04);
        b = im_size * (gamma(2.2, 0.6) / 10.0 + 0.04);
        mask = add_ellipse(mask, x, y, uniform() * math.pi, a, b);

    return (mask, ratio);

def rand_nuclei(im_size):
    mask, ratio = rand_mask(im_size);
    im = np.zeros(shape=(im_size, im_size, 3), dtype=np.uint8);
    im[:, :, 0] = 255 * (mask) * (0.22 + uniform(-0.01, 0.01));
    im[:, :, 1] = 255 * (mask) * (0.10 + uniform(-0.01, 0.01));
    im[:, :, 2] = 255 * (mask) * (0.30 + uniform(-0.01, 0.01));
    im[:, :, 0] = im[:, :, 0] + 255 * (1 - mask) * (0.82 + uniform(-0.01, 0.01));
    im[:, :, 1] = im[:, :, 1] + 255 * (1 - mask) * (0.35 + uniform(-0.01, 0.01));
    im[:, :, 2] = im[:, :, 2] + 255 * (1 - mask) * (0.60 + uniform(-0.01, 0.01));
    return (im, ratio);

#im, ratio = rand_nuclei(50);
#toimage(im).show()
#print ratio;

