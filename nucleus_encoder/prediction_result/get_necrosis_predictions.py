import numpy as np

PS = 200;

def append_necrosis_txt(pred_file, necrosisf):
    prediction = np.load(pred_file).reshape(-1, PS, PS);
    prob = prediction[:, 75:125, 75:125].mean(axis=(1,2));
    fid = open(necrosisf, 'a');
    for i in range(prob.shape[0]):
        fid.write('{}\n'.format(prob[i]));
    fid.close();

#necrosisf = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/visual_alt44/necrosis.txt';
#pred_file = 'result-pred_deep_segmentation_deconv_necrosis_test_alt1_foldid-0_8-2-highlym_test_larger_patches_pred.npy';
#append_necrosis_txt(pred_file, necrosisf);
#pred_file = 'result-pred_deep_segmentation_deconv_necrosis_test_alt2_foldid-0_8-2-nonhighlym_test_larger_patches_pred.npy';
#append_necrosis_txt(pred_file, necrosisf);

necrosisf = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/visual_alt45/necrosis.txt';
pred_file = 'result-pred_deep_segmentation_deconv_necrosis_test_alt3_foldid-0_MERGED-Anne_patches_Aug15.txt_larger_patches_pred.npy';
append_necrosis_txt(pred_file, necrosisf);

