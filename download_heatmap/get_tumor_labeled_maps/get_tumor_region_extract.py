# inputs: svs_name, username, image_path, width, height, patch_heat_pixel_n, sample_rate
# not tested

from __future__ import division, print_function
import numpy as np
import os
import sys
import cv2


def get_tumor_region_extract(svs_name, username, image_path, width, height, patch_heat_pixel_n, sample_rate):
    labs = np.floor(patch_heat_pixel_n/2 - 0.5)
    labe = np.floor(patch_heat_pixel_n/2)

    im = cv2.imread(image_path)
    im_height = im.shape[0]
    im_width = im.shape[1]

    im_temp = im.copy()
    im_temp[im == 255] = 0
    ys, xs = np.where(im_temp == 0)

    rp = np.random.permutation(len(xs))
    xs = xs[rp[range(0, len(xs), sample_rate)]]
    ys = ys[rp[range(0, len(ys), sample_rate)]]

    fid = open('tumor_image_to_extract/' + svs_name + '.' + username + '.txt', 'w')
    count_neg = 0
    count_pos = 0
    heatmap_ratio = width/im_width
    lab_width = int(100/heatmap_ratio)
    labs_center = np.floor(lab_width/2 - 0.5)
    labe_center = np.floor(lab_width/2)

    for i in range(len(xs)):
        if (xs[i] - labs < 1) or (ys[i] - labs < 1) or (xs[i] + labe > im_width) or (ys[i] + labe > im_height):
            continue

        lab_patch = im[ys[i] - labs:ys[i] + labe, xs[i] - labs:xs[i] + labe]
        lab_center_100 = im[ys[i] - labs_center:ys[i] + labe_center, xs[i] - labs_center:xs[i] + labe_center]

        # generate patch label from the patch's heatmap
        threshold = 0.5
        label = -1
        label_0 = np.sum(lab_center_100 == 0)
        label_1 = np.sum(lab_center_100 == 255)

        if label_0/(lab_width*lab_width) > threshold:
            label = 0
            count_neg += 1
        elif label_1/(lab_width*lab_width) > threshold:
            label = 1
            count_pos += 1

        # only save the heatmaps and labels if they are Neg/Pos patches, unknown patches will be saved
        if label >= 0:
            fid.write('{},{},{},{},{}\n'.format(int((xs[i]-labs-1)/im_width*width+1),
                                              int((ys[i]-labs-1)/im_height*height+1),
                                              int((xs[i] + labe) / im_width * width + 1),
                                              int((ys[i] + labe) / im_height * height + 1),
                                              label))

            patch_name = 'tumor_ground_truth/{}-{}-{}-{}-{}.png'.format(svs_name,
                                                                        int((xs[i] - labs-1)/im_width*width+1),
                                                                        int((ys[i] - labs-1) / im_height * height + 1),
                                                                        int((xs[i] + labe) / im_width * width + 1),
                                                                        int((ys[i] + labe) / im_height * height + 1))
            cv2.imwrite(patch_name, lab_patch)

        with open('tumor_ground_truth/label_summary.txt', 'a') as f:
            f.write('%s %d %d [svs, #Neg, #Pos]\n'.format(svs_name, count_neg, count_pos))

        fid.close()
