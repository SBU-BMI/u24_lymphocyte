# Eg
# cancer_type:    'luad'
# parent_out_dir (output directory): '/data07/shared/lehhou/lym_outputs/csv'
# src_dir (source directory):        '/data08/shared/lehhou/active_learning_osprey'

import os
import skimage
import skimage.io

def get_patch_til_svs_file_singlefile_wrap(cancer_type, parent_out_dir, src_dir):


    # parent_out_dir = '/data07/shared/lehhou/lym_outputs/csv';
    out_dir = os.path.join(parent_out_dir, cancer_type);
    if (os.path.isdir(out_dir) == 0):
        os.makedirs(out_dir);

    src_img_dir = src_dir + '/rates-' + cancer_type + '-all-auto';
    # src_img_dir = cancertype_path;
    # bin_src_imgs = dir(fullfile(src_img_dir, '*automatic.png'));
    # bin_src_imgs = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    bin_src_imgs = [f for f in os.listdir(src_img_dir) if f.endswith('automatic.png')]

    # parpool(12);
    for i_img in range(len(bin_src_imgs)):
        slide_name = bin_src_imgs[i_img][5:-14];
        csv_path = os.path.join(out_dir, slide_name + '.csv');
        info_path = os.path.join(out_dir, slide_name + '.info');
        fileID = open(csv_path, 'w');
        infoID = open(info_path, 'w');

        bin_img_name = bin_src_imgs[i_img][0:-4] + '_thres.png';
        real_img_name = bin_src_imgs[i_img];
        real_img_path = os.path.join(src_img_dir, real_img_name);
        bin_img_path = os.path.join(src_img_dir, bin_img_name);

        real_img = skimage.io.imread(real_img_path);
        bin_img = skimage.io.imread(bin_img_path);

        width = real_img.shape[1];
        height = real_img.shape[0];

        n_tissue = 0;
        n_til = 0;

        for iH in range(height):
            for iW in range(width):
                # real value
                real_value = float(real_img[iH, iW, 0]) / 255.0;

                # bin value
                bin_value = 0;
                # if this is pos tile
                if (bin_img[iH, iW, 0] > bin_img[iH, iW, 2]):
                    bin_value = 1;
                    n_til = n_til + 1;
                    n_tissue = n_tissue + 1;

                # if this is a tissue tile
                if (bin_img[iH, iW, 2] > 128):
                    n_tissue = n_tissue + 1;

                fileID.write('{},{},{},{:.4f}\n'.format(iH, iW, bin_value, real_value));

        infoID.write('{},{}\n'.format('tissue_number', n_tissue))
        infoID.write('{},{}\n'.format('til_number', n_til));

        fileID.close();
        infoID.close();

    # delete(gcp);


if __name__== "__main__":
    get_patch_til_svs_file_singlefile_wrap('prad', 'res', 'data')
