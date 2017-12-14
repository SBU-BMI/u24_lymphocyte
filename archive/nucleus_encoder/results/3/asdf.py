import numpy as np
import h5py

X = np.zeros((2, 3, 32, 32));
fn = 'TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_appMag_20_10_8-seg.h5';
f = h5py.File('data/nuclei_image/train/' + fn, 'r');

X[0, :, :, :] = f['/data'][...][:, 0 : 3, 9 : 41, 9 : 41];

