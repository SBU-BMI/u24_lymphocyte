function get_modified_heatmap(svs_name, width, height, username, weight_file, mark_file, pred_file)
% generate human made binary map

[pred, pred_binary, necr, modification, tumor, patch_size, ~, ~, ~, ~, ~] = ...
    get_labeled_im(weight_file, mark_file, pred_file, width, height);

total = size(modification,1)*size(modification,2);
lym = sum(modification(:)>0.5);

im = zeros(size(modification,1), size(modification,2), 3, 'uint8');
im(:, :, 1) = 64*uint8(pred_binary+1);
im(:, :, 1) = im(:, :, 1) + 255*uint8(modification==255);
im(:, :, 1) = im(:, :, 1) - 255*uint8(modification==0);
im(:, :, 2) = 255*uint8(tumor>0.5);

fid = fopen(['heatmaps/', svs_name, '.', username, '.csv'], 'w');
[x, y] = find(modification==0 | modification==255);
fprintf(fid, 'X0,Y0,X1,Y1,PredProb,PredBinary,Corrected\n');
for i = 1:length(x)
    fprintf(fid, '%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d\n', ...
        (x(i)-1)*patch_size, (y(i)-1)*patch_size, x(i)*patch_size, y(i)*patch_size, ...
        pred(x(i),y(i)), pred_binary(x(i),y(i)), uint8(modification(x(i),y(i))>127));
end
fclose(fid);

im = permute(im, [2, 1, 3]);
imwrite(im, ['heatmaps/', svs_name, '.', username, '.png']);

