import sys
import os
import numpy as np
from PIL import Image

from external_model import load_external_model, pred_by_external_model

APS = 100;
BatchSize = 96;

TileFolder = sys.argv[1] + '/';
CNNModel = sys.argv[2];
heat_map_out = sys.argv[3];


def whiteness(png):
    wh = (np.std(png[:,:,0].flatten()) + np.std(png[:,:,1].flatten()) + np.std(png[:,:,2].flatten())) / 3.0;
    return wh;


def load_data(todo_list, rind):
    X = np.zeros(shape=(BatchSize*40, 3, APS, APS), dtype=np.float32);
    inds = np.zeros(shape=(BatchSize*40,), dtype=np.int32);
    coor = np.zeros(shape=(20000000, 2), dtype=np.int32);

    xind = 0;
    lind = 0;
    cind = 0;
    for fn in todo_list:
        lind += 1;
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

                if (whiteness(png[y:y+APS, x:x+APS, :]) >= 12):
                    X[xind, :, :, :] = png[y:y+APS, x:x+APS, :].transpose();
                    inds[xind] = rind;
                    xind += 1;

                coor[cind, 0] = np.int32(x_off + (x + APS/2) * svs_pw / png_pw);
                coor[cind, 1] = np.int32(y_off + (y + APS/2) * svs_pw / png_pw);
                cind += 1;
                rind += 1;

        if xind >= BatchSize:
            break;

    X = X[0:xind];
    inds = inds[0:xind];
    coor = coor[0:cind];

    return todo_list[lind:], X, inds, coor, rind;


def val_fn_epoch_on_disk(classn, model):
    all_or = np.zeros(shape=(20000000, classn), dtype=np.float32);
    all_inds = np.zeros(shape=(20000000,), dtype=np.int32);
    all_coor = np.zeros(shape=(20000000, 2), dtype=np.int32);
    rind = 0;
    n1 = 0;
    n2 = 0;
    n3 = 0;
    todo_list = os.listdir(TileFolder);
    while len(todo_list) > 0:
        todo_list, inputs, inds, coor, rind = load_data(todo_list, rind);
        if len(inputs) == 0:
            break;

        output = pred_by_external_model(model, inputs);

        all_or[n1:n1+len(output)] = output;
        all_inds[n2:n2+len(inds)] = inds;
        all_coor[n3:n3+len(coor)] = coor;
        n1 += len(output);
        n2 += len(inds);
        n3 += len(coor);

    all_or = all_or[:n1];
    all_inds = all_inds[:n2];
    all_coor = all_coor[:n3];
    return all_or, all_inds, all_coor;


def split_validation(classn):
    model = load_external_model(CNNModel)

    # Testing
    Or, inds, coor = val_fn_epoch_on_disk(classn, model);
    Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32);
    Or_all[inds] = Or[:, 0];

    fid = open(TileFolder + '/' + heat_map_out, 'w');
    for idx in range(0, Or_all.shape[0]):
        fid.write('{} {} {}\n'.format(coor[idx][0], coor[idx][1], Or_all[idx]));
    fid.close();

    return;


def main():
    if not os.path.exists(TileFolder):
        exit(0);

    classes = ['Lymphocytes'];
    classn = len(classes);
    sys.setrecursionlimit(10000);

    split_validation(classn);
    print('DONE!');


if __name__ == "__main__":
    main();
