function get_lym_infiltration_auto(svs_name, width, height, username, weight_file, mark_file, pred_file, white_file)
% generate probability map

if exist(['rates/rate.', svs_name, '.', username, '.png'], 'file') > 0
    return;
end

[pred, necr, truth, tumor, ~, ~, ~, ~, ~] = get_labeled_im(weight_file, mark_file, pred_file, width, height);
[whiteness, blackness, redness] = get_whiteness_im(white_file);
[pred, necr, truth, tumor, whiteness] = region_filter(pred, necr, truth, tumor, whiteness, svs_name);

im = zeros(size(pred,1), size(pred,2), 3, 'uint8');
im(:, :, 1) = 255*pred .* double(blackness>30) .* double(redness<0.15);
im(:, :, 2) = 255*necr;
im(:, :, 3) = 255*uint8(get_tissue_map(whiteness));
im = permute(im, [2, 1, 3]);
imwrite(im, ['rates/rate.', svs_name, '.', username, '.png']);

