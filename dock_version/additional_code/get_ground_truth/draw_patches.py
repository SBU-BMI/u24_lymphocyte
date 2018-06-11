import numpy as np
import h5py
import openslide
import sys
import os
from PIL import Image

labeled_txt = sys.argv[1];
output_folder = './patches/' + sys.argv[1].split('/')[-1];
svs_full_path = sys.argv[2];
patch_size_20X = 500;

try:
    mpp_w_h = os.popen('bash ../util/get_mpp_w_h.sh {}'.format(svs_full_path)).read();
    if len(mpp_w_h.split()) != 3:
        print '{}: mpp_w_h wrong'.format(svs_full_path);
        exit(1);

    mpp = float(mpp_w_h.split()[0]);
    width = int(mpp_w_h.split()[1]);
    height = int(mpp_w_h.split()[2]);
    if (mpp < 0.01 or width < 1 or height < 1):
        print '{}: mpp, width, height wrong'.format(svs_full_path);
        exit(1);
except:
    print '{}: exception caught'.format(svs_full_path);
    exit(1);

mag = 10.0 / mpp;
pw = int(patch_size_20X * mag / 20);

if not os.path.exists(output_folder):
    os.mkdir(output_folder);
fid = open(output_folder + '/label.txt', 'w');

obj_ids = 0;
lines = [line.rstrip('\n') for line in open(labeled_txt)];
for _, line in enumerate(lines):
    fields = line.split('\t');
    iid = fields[0];
    width = float(fields[6]);
    height = float(fields[7]);
    x = int(float(fields[2]) * width);
    y = int(float(fields[3]) * height);
    pred = float(fields[4]);
    label = int(fields[5]);
    fname = output_folder + '/{}.png'.format(obj_ids);

    os.system('bash ../util/save_tile.sh {} {} {} {} {} {}'.format(slide_name, (x-pw/2), (y-pw/2), pw, pw, fname));
    patch = Image.open(fname).resize((patch_size_20X, patch_size_20X), Image.ANTIALIAS);
    patch.save(fname);

    fid.write('{}.png {} {} {} {} {:.3f}\n'.format(obj_ids, label, iid, x, y, pred));

    obj_ids += 1;

fid.close();

