import numpy as np
from sklearn.metrics import roc_auc_score

vgg_lym_file = '/data08/shared/lehhou/nucleus_encoder/prediction_result/intermediate_visualization/vgg16_train_on_highlym/output.txt';
vgg_nonnecrotis_file = '/data08/shared/lehhou/nucleus_encoder/prediction_result/intermediate_visualization/vgg16_train_on_necrotis/output.txt';
ae_lym_file = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/visual_alt27/output.txt';
ae_nonnecrotis_file = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/visual_alt35/output.txt';
truth_file = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/visual_alt35/truth.txt';

vgg_lym = np.loadtxt(vgg_lym_file);
vgg_nonnecrotis = np.loadtxt(vgg_nonnecrotis_file);
ae_lym = np.loadtxt(ae_lym_file);
ae_nonnecrotis = np.loadtxt(ae_nonnecrotis_file);
groundtruth = np.loadtxt(truth_file);

sum_weight0 = [0.4, 0.4, 0.1, 0.1];
sum_weight1 = [0.5, 0.3, 0.1, 0.1];
sum_weight2 = [0.5, 0.5, 0.0, 0.0];
sum_weight3 = [0.65, 0.35, 0.0, 0.0];

score0 = ae_lym*sum_weight0[0] + vgg_lym*sum_weight0[1] + ae_nonnecrotis*sum_weight0[2] + vgg_nonnecrotis*sum_weight0[3];
score1 = ae_lym*sum_weight1[0] + vgg_lym*sum_weight1[1] + ae_nonnecrotis*sum_weight1[2] + vgg_nonnecrotis*sum_weight1[3];
score2 = ae_lym*sum_weight2[0] + vgg_lym*sum_weight2[1] + ae_nonnecrotis*sum_weight2[2] + vgg_nonnecrotis*sum_weight2[3];
score3 = ae_lym*sum_weight3[0] + vgg_lym*sum_weight3[1] + ae_nonnecrotis*sum_weight3[2] + vgg_nonnecrotis*sum_weight3[3];

geo_score0 = np.power(np.power(ae_lym,4) * np.power(vgg_lym,4) * np.power(ae_nonnecrotis,1) * np.power(vgg_nonnecrotis,1), 0.1);
geo_score1 = np.power(np.power(ae_lym,5) * np.power(vgg_lym,3) * np.power(ae_nonnecrotis,1) * np.power(vgg_nonnecrotis,1), 0.1);
geo_score2 = np.power(np.power(ae_lym,5) * np.power(vgg_lym,5) * np.power(ae_nonnecrotis,0) * np.power(vgg_nonnecrotis,0), 0.1);
geo_score3 = np.power(np.power(ae_lym,6.5) * np.power(vgg_lym,3.5) * np.power(ae_nonnecrotis,0) * np.power(vgg_nonnecrotis,0), 0.1);

har_score0 = 10/(1/ae_lym/4 + 1/vgg_lym/4 + 1/ae_nonnecrotis/1 + 1/vgg_nonnecrotis/1);
har_score1 = 10/(1/ae_lym/5 + 1/vgg_lym/3 + 1/ae_nonnecrotis/1 + 1/vgg_nonnecrotis/1);
har_score2 = 10/(1/ae_lym/5 + 1/vgg_lym/5);
har_score3 = 10/(1/ae_lym/6.5 + 1/vgg_lym/3.5);

auc_vgg_lym = roc_auc_score(groundtruth, vgg_lym);
auc_ae_lym = roc_auc_score(groundtruth, ae_lym);
auc_vgg_nonne = roc_auc_score(groundtruth, vgg_nonnecrotis);
auc_ae_nonne = roc_auc_score(groundtruth, ae_nonnecrotis);

auc0 = roc_auc_score(groundtruth, score0);
auc1 = roc_auc_score(groundtruth, score1);
auc2 = roc_auc_score(groundtruth, score2);
auc3 = roc_auc_score(groundtruth, score3);

auc_geo0 = roc_auc_score(groundtruth, geo_score0);
auc_geo1 = roc_auc_score(groundtruth, geo_score1);
auc_geo2 = roc_auc_score(groundtruth, geo_score2);
auc_geo3 = roc_auc_score(groundtruth, geo_score3);

auc_har0 = roc_auc_score(groundtruth, har_score0);
auc_har1 = roc_auc_score(groundtruth, har_score1);
auc_har2 = roc_auc_score(groundtruth, har_score2);
auc_har3 = roc_auc_score(groundtruth, har_score3);

print ('VGG_Lym\t\tAE_Lym\t\tVGG_Necr\tAE_Necr');
print ('{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(auc_vgg_lym,auc_ae_lym,auc_vgg_nonne,auc_ae_nonne));
print ('Arth0\t\tArth1\t\tArth2\t\tArth3');
print ('{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(auc0, auc1, auc2, auc3));
print ('Geo0\t\tGeo1\t\tGeo2\t\tGeo3');
print ('{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(auc_geo0, auc_geo1, auc_geo2, auc_geo3));
print ('Har0\t\tHar1\t\tHar2\t\tHar3');
print ('{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(auc_har0, auc_har1, auc_har2, auc_har3));





