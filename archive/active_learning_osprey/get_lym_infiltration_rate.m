function get_lym_infiltration_rate(svs_name, width, height, username, weight_file, mark_file, pred_file, white_file)
% generate human made binary map

if exist(['rates/rate.', svs_name, '.', username, '.png'], 'file') > 0
    return;
end

[pred, necr, truth, tumor, ~, ~, ~, ~, ~] = get_labeled_im(weight_file, mark_file, pred_file, width, height);
[whiteness, blackness, redness] = get_whiteness_im(white_file);
[pred, necr, truth, tumor, whiteness] = region_filter(pred, necr, truth, tumor, whiteness, svs_name);

fid = fopen('rates/infiltration_rate.txt', 'a');
total = size(truth,1)*size(truth,2);
lym = sum(truth(:)>0.5);
tissue = get_tissue_map(whiteness);
nonwhite = sum(tissue(:)>0);
fprintf(fid, '%s\t%s\t%.6f\n', svs_name, username, double(lym)/nonwhite);
fclose(fid);

im = zeros(size(truth,1), size(truth,2), 3, 'uint8');
im(:, :, 1) = 255*uint8(truth>0.5);
im(:, :, 2) = 255*uint8(tumor>0.5);
im(:, :, 3) = 255*uint8(tissue>0);
im(:, :, 3) = im(:, :, 3) .* uint8(truth<0.5);

im = permute(im, [2, 1, 3]);
imwrite(im, ['rates/rate.', svs_name, '.', username, '.png']);

