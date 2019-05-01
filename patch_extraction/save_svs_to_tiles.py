import numpy as np
import openslide
import sys
import os
from PIL import Image

slide_name = sys.argv[2] + '/' + sys.argv[1];
output_folder = sys.argv[3] + '/' + sys.argv[1];
patch_size_20X = 1000;

fdone = '{}/extraction_done.txt'.format(output_folder);
if os.path.isfile(fdone):
    print 'fdone {} exist, skipping'.format(fdone);
    exit(0);

print 'extracting {}'.format(output_folder);

if not os.path.exists(output_folder):
    os.mkdir(output_folder);

try:
    oslide = openslide.OpenSlide(slide_name);
#    mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"]);
    elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
        mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
    else:
        print('[WARNING] mpp value not found. Assuming it is 40X with mpp=0.254!', slide_name);
        mag = 10.0 / float(0.254); 
    pw = int(patch_size_20X * mag / 20);
    width = oslide.dimensions[0];
    height = oslide.dimensions[1];
except:
    print '{}: exception caught'.format(slide_name);
    exit(1);


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
        patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
        patch = patch.resize((patch_size_20X * pw_x / pw, patch_size_20X * pw_y / pw), Image.ANTIALIAS);
        fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_20X);
        patch.save(fname);

open(fdone, 'w').close();

