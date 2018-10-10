from PIL import Image
import sys
import numpy as np
import cv2
from scipy import misc

thumb_nail = np.array(Image.open(sys.argv[1]).convert('RGB'))
tumor_fill = np.array(Image.open(sys.argv[2]).convert('L'))
tumor_fill = misc.imresize(tumor_fill, (thumb_nail.shape[0], thumb_nail.shape[1]))

tumor_contour = cv2.Canny(tumor_fill, 1, 1)
tumor_contour = cv2.dilate(tumor_contour, np.ones((3,3)), iterations=3)

thumb_nail[tumor_contour>0, 1] = 255

Image.fromarray(thumb_nail).save(sys.argv[3])
