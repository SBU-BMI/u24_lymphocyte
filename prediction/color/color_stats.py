import sys
import os
import numpy as np
import time
from PIL import Image

APS = 100;
TileFolder = sys.argv[1] + '/';
heat_map_out = sys.argv[2];

def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def blackness(png):
    bk = np.mean(png);
    return bk;


def redness(png):
    rd = np.mean((png[:,:,0] >= 190) * (png[:,:,1] <= 100) * (png[:,:,2] <= 100));
    return rd;


def load_data():
    X = np.zeros(shape=(1000000, 3), dtype=np.float32);
    coor = np.zeros(shape=(1000000, 2), dtype=np.int32);

    ind = 0;
    for fn in os.listdir(TileFolder):
        full_fn = TileFolder + '/' + fn;
        if not os.path.isfile(full_fn):
            continue;
        if len(fn.split('_')) < 4:
            continue;

        x_off = float(fn.split('_')[0]);
        y_off = float(fn.split('_')[1]);
        svs_pw = float(fn.split('_')[2]);
        png_pw = float(fn.split('_')[3].split('.png')[0]);

        png = np.array(Image.open(full_fn).convert('RGB'));
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue;
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue;
                X[ind, 0] = whiteness(png[y:y+APS, x:x+APS, :]);
                X[ind, 1] = blackness(png[y:y+APS, x:x+APS, :]);
                X[ind, 2] = redness(png[y:y+APS, x:x+APS, :]);
                coor[ind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                coor[ind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                ind += 1;

    X = X[0:ind];
    coor = coor[0:ind];

    return X, coor;


def split_validation():
    Wh, coor = load_data();

    fid = open(TileFolder + '/' + heat_map_out, 'w');
    for idx in range(0, Wh.shape[0]):
        fid.write('{} {} {} {} {}\n'.format(coor[idx][0], coor[idx][1], Wh[idx][0], Wh[idx][1], Wh[idx][2]));
    fid.close();


def main():
    split_validation();


if __name__ == "__main__":
    main();

