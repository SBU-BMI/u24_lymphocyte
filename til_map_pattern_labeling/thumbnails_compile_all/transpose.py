from scipy import misc
import sys
import numpy as np

im = misc.imread(sys.argv[1])
im = np.swapaxes(im, 0, 1)[:, ::-1, ...]
misc.imsave(sys.argv[2], im)
