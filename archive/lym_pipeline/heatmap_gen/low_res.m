function low_res(fname)

[x, y, p, w] = textread(fname, '%d%d%f%f');
step = double(max(x(:))) / length(unique(x(:)));

x = round((x+step/2) / step);
y = round((y+step/2) / step);

imp = zeros(length(unique(x(:))), length(unique(y(:))));
for iter = 1:length(x)
    i = x(iter);
    j = y(iter);
    imp(x(iter), y(iter)) = p(iter);
end

imn = zeros(length(unique(x(:))), length(unique(y(:))));
for iter = 1:length(x)
    i = x(iter);
    j = y(iter);
    imn(x(iter), y(iter)) = w(iter);
end

fid = fopen([fname, '.low_res'], 'w');
for i = 1:floor(size(imp, 1)/4)
    for j = 1:floor(size(imp, 2)/4)
        p_val = max(max(imp((i-1)*4+1:i*4, (j-1)*4+1:j*4)));
        w_val = min(min(imn((i-1)*4+1:i*4, (j-1)*4+1:j*4)));
        fprintf(fid, '%d %d %f %f\n', round((i-0.5)*step*4.0), round((j-0.5)*step*4.0), p_val, w_val);
    end
end
fclose(fid);

