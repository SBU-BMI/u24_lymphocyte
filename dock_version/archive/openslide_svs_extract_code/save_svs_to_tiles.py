import numpy as np
import h5py
import sys
import os
from PIL import Image

slide_name = sys.argv[2] + '/' + sys.argv[1];
output_folder = sys.argv[3] + '/' + sys.argv[1];
patch_size_20X = 1000;

if not os.path.exists(output_folder):
    os.mkdir(output_folder);

try:
    mpp_w_h = os.popen('bash get_mpp_w_h.sh {}'.format(slide_name)).read();
    if len(mpp_w_h.split()) != 3:
        print '{}: mpp_w_h wrong'.format(slide_name);
        exit(1);

    mpp = float(mpp_w_h.split()[0]);
    width = int(mpp_w_h.split()[1]);
    height = int(mpp_w_h.split()[2]);
    if (mpp < 0.01 or width < 1 or height < 1):
        print '{}: mpp, width, height wrong'.format(slide_name);
        exit(1);
    mag = 10.0 / mpp;
except:
    print '{}: exception caught'.format(slide_name);
    exit(1);

pw = int(patch_size_20X * mag / 20);

print slide_name, width, height;

for x in range(1, width, pw):
    for y in range(1, height, pw):
        if x + pw > width:
            pw_x = width - x;
        else:
            pw_x = pw;
        if y + pw > height:
            pw_y = height - y;
        else:
            pw_y = pw;
        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_20X);
        os.system('bash save_tile.sh {} {} {} {} {} {}'.format(slide_name, x, y, pw_x, pw_y, fname));
        patch = Image.open(fname).resize((patch_size_20X * pw_x / pw, patch_size_20X * pw_y / pw), Image.ANTIALIAS);
        patch.save(fname);

