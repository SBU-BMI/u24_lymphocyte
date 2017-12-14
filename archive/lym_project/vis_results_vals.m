function vis_results_vals()

pred1 = load('./visual/deep_conv_classification_alt36_deploy.py/output.txt');
pred1 = tiedrank(pred1) / length(pred1);
pred2 = load('./visual/deep_conv_classification_alt35_deploy.py/output.txt');
pred2 = tiedrank(pred2) / length(pred2);
lab = load('./visual/deep_conv_classification_alt35_deploy.py/truth.txt');
crop_im = zeros(size(pred1, 1), 300, 300, 3, 'uint8');
for i = 1 : size(pred1, 1)
    im = imread(sprintf('./visual/deep_conv_classification_alt35_deploy.py/case_%d.png', i));
    im(100:200, 100, 2) = 255;
    im(100:200, 200, 2) = 255;
    im(100, 100:200, 2) = 255;
    im(200, 100:200, 2) = 255;
    crop_im(i, :, :, :) = im;
end

fprintf('Image read.\n');
[~, ind] = sortrows(round(pred1 * 10));
lab = lab(ind(end:-1:1), :);
pred1 = pred1(ind(end:-1:1), :);
pred2 = pred2(ind(end:-1:1), :);
crop_im = crop_im(ind(end:-1:1), :, :, :);

fprintf('Numbers ranked.\n');
for i = 1 : size(crop_im, 1)
    subplot(1, 4, [1, 2, 3]), imshow(squeeze(crop_im(i, :, :, :)));
    subplot(1, 4, 4),         bar_pred_truth_vals(lab(i, :), pred1(i, :), pred2(i, :));
    frame = getframe(gcf);
    imwrite(frame.cdata, sprintf('./to_vision_cs/%d.png', i));
    close all;
end

