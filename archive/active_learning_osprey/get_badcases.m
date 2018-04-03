function get_badcases(svs_name, width, height, username, weight_file, mark_file, pred_file)

[pred, necr, truth, tumor, ~, calc_width, calc_height, tot_width, tot_height] = get_labeled_im(weight_file, mark_file, pred_file, width, height);
[pred, necr, truth, tumor, ~] = region_filter(pred, necr, truth, tumor, pred, svs_name);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Count all tissue
%false_neg = truth .* uint8(pred < 0.98);
%false_pos = (1-truth) .* uint8(pred > 0.02);

% Only count tissue circled in tumor
false_neg = truth .* uint8(pred <= 1.0) .* uint8(tumor > 0.5);
false_pos = (1-truth) .* uint8(pred >= 0.0) .* uint8(tumor > 0.5);

badcases = false_neg + false_pos;
[xs, ys] = find(badcases > 0.5);

fid = fopen(['logs/log.', svs_name, '.', username, '.txt'], 'w');
for i = 1:length(xs)
    fprintf(fid, '%s\t%s\t%.8f\t%.8f\t%.3f\t%d\t%d\t%d\t%d\t%d\n', svs_name, username, ...
        (xs(i)-0.5) / size(pred,1), (ys(i)-0.5) / size(pred,2), ...
        pred(xs(i), ys(i)), truth(xs(i), ys(i)), ...
        calc_width, calc_height, tot_width, tot_height);
end
fclose(fid);

