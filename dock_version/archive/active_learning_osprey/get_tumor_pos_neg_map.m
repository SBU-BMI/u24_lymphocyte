function get_tumor_pos_neg_map(svs_name, width, height, username, weight_file, mark_file, pred_file, white_file)

[pred, necr, truth, tumor, notum, calc_width, calc_height, tot_width, tot_height] = get_labeled_im_high_res(weight_file, mark_file, pred_file, width, height);
[whiteness, blackness, redness] = get_whiteness_im(white_file);
[pred, necr, truth, tumor, whiteness] = region_filter(pred, necr, truth, tumor, whiteness, svs_name);

im = zeros(size(truth,1), size(truth,2), 3, 'uint8');
im(:, :, 1) = 255*uint8(tumor>0.5);
im(:, :, 2) = 255*uint8(notum>0.5) .* uint8(tumor<0.5);
im = permute(im, [2, 1, 3]);
imwrite(im, ['tumor_nontumor_maps/map.', svs_name, '.', username, '.png']);

fid = fopen('tumor_nontumor_maps/meta_data.txt', 'a');
fprintf(fid, 'map.%s.%s.png\t%.12f\n', svs_name, username, calc_height/size(im,1));
fclose(fid);
