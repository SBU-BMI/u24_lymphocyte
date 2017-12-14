function get_patch_coordinates(svs_name, username, image_path, pred_file, tot_width, tot_height)

annotat = imread(['./annotation_maps/', svs_name, '----', username, '.png']);
neg = (annotat <= 10);
pos = (annotat >= 250);

allcases = neg + pos;
[xs, ys] = find(allcases > 0.5);

fid = fopen(['./patch_coordinates/', svs_name, '----', username, '.txt'], 'w');
for i = 1:length(xs)
    fprintf(fid, '%s\t%s\t%.8f\t%.8f\t%.3f\t%d\t%d\t%d\t%d\t%d\n', svs_name, username, ...
        (xs(i)-0.5) / size(pred,1), (ys(i)-0.5) / size(pred,2), ...
        pred(xs(i), ys(i)), truth(xs(i), ys(i)), ...
        calc_width, calc_height, tot_width, tot_height);
end
fclose(fid);

