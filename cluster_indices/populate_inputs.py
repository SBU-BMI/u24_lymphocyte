import os
from get_patch_til_svs_file_wrap import get_patch_til_svs_file_singlefile_wrap


processedImages = '/data08/shared/lehhou/active_learning_osprey';
outputs = 'inputs';
cancer_types = ['prad'];

for ctype in cancer_types:
    os.makedirs(os.path.join(outputs, ctype));
    get_patch_til_svs_file_singlefile_wrap(ctype, outputs, processedImages);

