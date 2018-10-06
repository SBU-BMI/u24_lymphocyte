###input: svs_name, image_path, pred_file, tot_width, tot_height

from __future__ import print_function, division
import os
import sys
import numpy as np
import sys
import pandas as pd
import cv2

svs_name = sys.argv[1]
image_path = sys.argv[2]
pred_file = sys.argv[3]
tot_width = int(sys.argv[4])
tot_height = int(sys.argv[5])

print(pred_file)

def xy_in_conf_interval(pred, low, high):
    if (low < 0):
        xs = []
        ys = []
        return []
    bcase = (pred > low).astype(np.uint8) * (pred < high).astype(np.uint8)
    xs, ys = np.where(bcase > 0.5)
    if len(xs) > 0:
        ind = np.floor(len(xs)*(np.random.rand() + 0.001)).astype(np.uint32)
        xs = xs[ind]
        ys = ys[ind]
    else:
        xs, ys = xy_in_conf_interval(pred, low - 0.05, high + 0.05)

    return xs, ys


df = pd.read_csv(pred_file, delimiter=' ', names=['x', 'y', 'lym', 'nec'])
x = np.array(df['x'])
y = np.array(df['y'])

calc_width = np.max(x) + np.min(x)
calc_height = np.max(y) + np.min(y)

im = cv2.imread(image_path) # this is BGR image
pred = im[:,:,2]/255.0  # read channel R

xs1, ys1 = xy_in_conf_interval(pred, 0.00, 0.10)
xs2, ys2 = xy_in_conf_interval(pred, 0.10, 0.20)
xs3, ys3 = xy_in_conf_interval(pred, 0.20, 0.3)
xs4, ys4 = xy_in_conf_interval(pred, 0.30, 0.4)
xs5, ys5 = xy_in_conf_interval(pred, 0.40, 0.5)
xs6, ys6 = xy_in_conf_interval(pred, 0.50, 0.6)
xs7, ys7 = xy_in_conf_interval(pred, 0.60, 0.7)
xs8, ys8 = xy_in_conf_interval(pred, 0.70, 0.8)
xs9, ys9 = xy_in_conf_interval(pred, 0.80, 0.9)
xs0, ys0 = xy_in_conf_interval(pred, 0.90, 1.0)

xs = [xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs0]
ys = [ys1, ys2, ys3, ys4, ys5, ys6, ys7, ys8, ys9, ys0]

with open('sample_list/' + svs_name + '.txt', 'w') as f:
    for i in range(len(xs)):
        f.write('{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}\n'.format((xs[i]-1)/pred.shape[0]*calc_width,
                                          (ys[i] - 1)/pred.shape[1]*calc_height,
                                          (xs[i] + 0)/pred.shape[0]*(calc_width),
                                          (ys[i] + 0)/pred.shape[1]*calc_height,
                                          pred[int(xs[i]), int(ys[i])]))
