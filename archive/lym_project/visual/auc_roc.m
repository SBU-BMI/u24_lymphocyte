function auc_roc()

truth = load('deep_conv_classification_alt48_luad10_skcm10_lr0_deploy.py/truth.txt');
p1 = load('deep_conv_classification_alt48_luad10_skcm10_lr0_deploy.py/output.txt');
p2 = load('/data08/shared/lehhou/nucleus_encoder/prediction_result/result-pred_vgg_foldid-0_20170321-172746.txt');

[x1, y1, ~, auc1] = perfcurve(truth, p1, 1);
[x2, y2, ~, auc2] = perfcurve(truth, p2, 1);

f = figure;
movegui(f, 'center');
plot(x1, y1, 'b');
hold on;
plot(x2, y2, 'r');

xlabel('False positive rate');
ylabel('True positive rate');
title('Receiver Operating Characteristic (ROC) curve');
legend(sprintf('Our method: %.4f Area Under ROC Curve', auc1), sprintf('VGG16 network: %.4f Area Under ROC Curve', auc2), 'Location', 'Southeast');


