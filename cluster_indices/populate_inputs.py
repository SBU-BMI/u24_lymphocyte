import os
import sys
from get_patch_til_svs_file_wrap import get_patch_til_svs_file_singlefile_wrap

cancer_types = ['read', 'uvm'];

processedImages = sys.argv[1];
outputs = sys.argv[2];

for ctype in cancer_types:
    os.makedirs(os.path.join(outputs, ctype));
    get_patch_til_svs_file_singlefile_wrap(ctype, outputs, processedImages);

