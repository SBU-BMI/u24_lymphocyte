import numpy as np
import os
from PIL import Image
import sys

img = sys.argv[1]
mean_threshold = int(sys.argv[2])
std_threshold = int(sys.argv[3])
png = np.array(Image.open(img).convert('RGB')).transpose();
_, w, h = png.shape
png = png[:, int(w/2) - 50: int(w/2) + 50, int(h/2) - 50: int(h/2) + 50]
mean = np.mean(png)
std = np.std(png)
file = 'isWhitePatch.txt'
if os.path.isfile(file):
    os.system('rm -rf ' + file)

# white patches for background: mean > 230, std < 5
# white patches for label 0/1: mean > 230, std < 20
if mean > mean_threshold and std < std_threshold:
    # this is white patch
    os.system('touch ' + file)
