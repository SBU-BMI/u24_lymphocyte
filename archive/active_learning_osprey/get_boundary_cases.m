function get_boundary_cases(svs_name, image_path, pred_file, tot_width, tot_height)

username = 'boundary_cases';

data = load(pred_file);
x = data(:, 1);
y = data(:, 2);
calc_width = max(x(:)) + min(x(:));
calc_height = max(y(:)) + min(y(:));

im = imread(image_path);
im = permute(im, [2, 1, 3]);
pred = double(im(:, :, 1)) / 255.0;

[xs1, ys1] = xy_in_conf_interval(pred, 0.08, 0.12);
[xs2, ys2] = xy_in_conf_interval(pred, 0.12, 0.16);
[xs3, ys3] = xy_in_conf_interval(pred, 0.16, 0.20);
[xs4, ys4] = xy_in_conf_interval(pred, 0.20, 0.30);
[xs5, ys5] = xy_in_conf_interval(pred, 0.30, 0.40);
[xs6, ys6] = xy_in_conf_interval(pred, 0.40, 0.50);
[xs7, ys7] = xy_in_conf_interval(pred, 0.50, 0.60);
[xs8, ys8] = xy_in_conf_interval(pred, 0.60, 0.70);
[xs9, ys9] = xy_in_conf_interval(pred, 0.70, 0.80);
[xs0, ys0] = xy_in_conf_interval(pred, 0.80, 0.90);
xs = [xs1; xs2; xs3; xs4; xs5; xs6; xs7; xs8; xs9; xs0];
ys = [ys1; ys2; ys3; ys4; ys5; ys6; ys7; ys8; ys9; ys0];

fid = fopen(['boundary_cases/log.', svs_name, '.', username, '.txt'], 'w');
for i = 1:length(xs)
    fprintf(fid, '%s\t%s\t%.8f\t%.8f\t%.3f\t%d\t%d\t%d\t%d\t%d\n', svs_name, username, ...
        (xs(i)-0.5) / size(pred,1), (ys(i)-0.5) / size(pred,2), ...
        pred(xs(i), ys(i)), -1, ...
        calc_width, calc_height, tot_width, tot_height);
end
fclose(fid);


function [xs, ys] = xy_in_conf_interval(pred, low, high)
if (low < 0)
    xs = [];
    ys = [];
    return;
end

bcase = uint8(pred > low) .* uint8(pred < high);
[xs, ys] = find(bcase > 0.5);
if length(xs) > 0
    ind = ceil(length(xs) * (rand()+0.001));
    xs = xs(ind);
    ys = ys(ind);
else
    [xs, ys] = xy_in_conf_interval(pred, low-0.05, high+0.05)
end

