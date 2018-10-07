# inputs: svs_name, width, height, username, weight_file, mark_file, pred_file
# need to use output of the file get_labeled_im.py to be written by Le Hou

# NEED testing...

import numpy as np
import cv2
import sys

svs_name = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
username = sys.argv[4]
weight_file = sys.argv[5]
mark_file = sys.argv[6]
pred_file = sys.argv[7]

pred, pred_binary, necr, modification, tumor, patch_size, _, _, _, _, _ = get_labeled_im(
                                            weight_file, mark_file, pred_file, width, height)

total = modification.shape[0]*modification.shape[1]
lym = np.sum(modification > 0.5)

im = np.zeros(modification.shape[0], modification.shape[1], 3).astype(np.uint8)

im[:, :, 0] = 64*(pred_binary + 1).astype(np.uint8)
im[:, :, 0] = im[:, :, 0] + 255*(modification == 255).astype(np.uint8)
im[:, :, 0] = im[:, :, 0] - 255*(modification == 0).astype(np.uint8)
im[:, :, 1] = 255*(tumor > 0.5).astype(np.uint8)


modification[modification == 0] = 255
x, y = np.where(modification == 255)

with open('modified_heatmaps/' + svs_name + '.' + username + '.csv', 'w') as f:
    f.write('X0,Y0,X1,Y1,PredProb,PredBinary,Corrected\n')
    for i in range(len(x)):
        f.write('{},{},{},{},{},{},{}'.format((x[i]-1)*patch_size, (y[i]-1)*patch_size, x[i]*patch_size, y[i]*patch_size,
            pred[x[i],y[i]], pred_binary[x[i],y[i]], int(modification[x[i],y[i]]>127)))

im = np.transpose(im, (2, 0, 1))
cv2.imwrite('modified_heatmaps/' + svs_name + '.' + username, im)




