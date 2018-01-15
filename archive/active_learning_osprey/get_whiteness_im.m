function [whiteness, blackness, redness] = get_whiteness_im(white_file)

data = load(white_file);
x = data(:, 1);
y = data(:, 2);
w = data(:, 3);
b = data(:, 4);
r = data(:, 5);

step = (double(min(x(:))) + double(max(x(:)))) / length(unique(x(:)));

x = round((x+step/2) / step);
y = round((y+step/2) / step);

whiteness = zeros(max(x(:)), max(y(:)));
blackness = zeros(max(x(:)), max(y(:)));
redness = zeros(max(x(:)), max(y(:)));

for iter = 1:length(x)
    whiteness(x(iter), y(iter)) = w(iter);
    blackness(x(iter), y(iter)) = b(iter);
    redness(x(iter), y(iter)) = r(iter);
end

