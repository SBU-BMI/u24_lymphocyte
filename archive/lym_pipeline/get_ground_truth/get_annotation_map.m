function get_annotation_map(svs_name, width, height, username, weight_file, mark_file, pred_file, white_file)
% generate human made binary map

[~, ~, annotat, ~, ~, ~, ~, ~, ~, ~] = get_modified_prediction(weight_file, mark_file, pred_file, width, height);
annotat = permute(annotat, [2, 1]);
imwrite(annotat, ['./annotation_maps/', svs_name, '----', username, '.png']);

