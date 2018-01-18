function [pred, necr, patch_size] = get_labeled_im(pred_file)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pred_data = load(pred_file);
x = int32(pred_data(:, 1));
y = int32(pred_data(:, 2));
l = pred_data(:, 3);
if size(pred_data, 2) > 3
    n = pred_data(:, 4);
else
    n = zeros(length(x), 1);
end
calc_width = max(x(:)) + min(x(:));
calc_height = max(y(:)) + min(y(:));
patch_size = (double(min(x(:))) + double(max(x(:)))) / length(unique(x(:)));

x = round((x+patch_size/2) / patch_size);
y = round((y+patch_size/2) / patch_size);

pred = zeros(max(x(:)), max(y(:)));
for iter = 1:length(x)
    pred(x(iter), y(iter)) = l(iter);
end

necr = zeros(max(x(:)), max(y(:)));
for iter = 1:length(x)
    necr(x(iter), y(iter)) = n(iter);
end

