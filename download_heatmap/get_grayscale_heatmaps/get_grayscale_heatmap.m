function get_modified_heatmap(svs_name, width, height, pred_file, color_file)

[pred, necr, patch_size] = get_labeled_im(pred_file);
[whiteness, blackness, redness] = get_whiteness_im(color_file);

im = zeros(size(pred,1), size(pred,2), 3, 'uint8');
im(:, :, 1) = 255 * pred .* double(blackness>30) .* double(redness<0.15);
im(:, :, 2) = 255 * necr;
im(:, :, 3) = 255 * uint8(get_tissue_map(whiteness));

im = permute(im, [2, 1, 3]);
imwrite(im, ['grayscale_heatmaps/', svs_name, '.png']);

