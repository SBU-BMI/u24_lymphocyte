# inputs: svs_name, username, width, height, mark_file
# not tested by 10/7/18


import numpy as np
from get_labeled_im_high_res import *
from get_tumor_region_extract import *
import sys
import cv2

svs_name = sys.argv[1]
username = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])
mark_file = sys.argv[5]


PosLabel = 'LymPos'
NegLabel = 'LymNeg'
HeatPixelSize = 4
# to downscale the heatmap 4 times from the size of the slide
# 1 pixel in the heatmap corresponds to 4 pixel in the slide
PatchHeatPixelN = 25
PatchSampleRate = 50

tumor = get_labeled_im_high_res(mark_file, width, height, HeatPixelSize, PosLabel, NegLabel)
image_path = 'tumor_heatmaps/{}.{}.png'.format(svs_name, username)
cv2.imwrite(image_path, np.transpose(tumor))

get_tumor_region_extract(svs_name, username, image_path, width, height, PatchHeatPixelN, PatchSampleRate)
