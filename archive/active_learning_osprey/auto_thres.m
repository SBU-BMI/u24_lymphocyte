function auto_thres(folder)

[svss, dthres] = textread('auto_thres.txt', '%s%f', 'delimiter', ' ');
cont = containers.Map(svss, dthres);

fid = fopen([folder, '/infiltration_rate.txt'], 'a');
files = dir([folder, '/rate.*.automatic.png']);
for f = files'
    fname = f.name;
    save_fn = [fname(1:end-4), '_thres.png'];

    svs_name = strsplit(fname, '.automatic.png');
    svs_name = strsplit(svs_name{1}, 'rate.');
    svs_name = svs_name{2};

    if isKey(cont, svs_name) < 0.5
        dt = 0.0;
    else
        dt = values(cont, {svs_name});
        dt = dt{1};
    end
    fprintf('dthres: %s %f\n', svs_name, dt);

    im = imread([folder, '/', fname]);
    lym = lym_magic_thres(im(:, :, 1), dt);
    nec = nec_magic_thres(im(:, :, 2));
    lym = lym .* uint8(nec < 1);

    im(:, :, 1) = lym .* uint8(im(:, :, 3) > 1);
    im(:, :, 3) = im(:, :, 3) .* uint8(lym<0.5);

    imwrite(im, [folder, '/', save_fn]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nlym = sum(sum(double(im(:, :, 1)>0.5)));
    ntissue = nlym + sum(sum(double(im(:, :, 3)>0.5)));

    fprintf(fid, '%s\t%s\t%.6f\n', svs_name, 'automatic_thres', nlym/ntissue);
end
fclose(fid);


function lym = lym_magic_thres(im, dt)

h = 57.528;
l = 57.528;
w = 0.0558333;
%h = 50.0;
%l = 50.0;
%w = 0.1;

h = h - dt*255;
l = l - dt*255;

im = smoothing(im, 0.01+2*w*w);
bw = bwlabel(im>=l, 4);
for b = 1:max(bw(:))
    region = (bw == b);
    if max(max(im(region))) < h
        bw(bw == b) = 0;
    end
end
lym = uint8(255*uint8(bw>0.5));


function nec = nec_magic_thres(im)

%h = 30.005085;
%l = 30.005085;
%w = 0.0558333;
h = 153.0;
l = 153.0;
w = 0.0;

im = smoothing(im, 0.01+2*w*w);
bw = bwlabel(im>=l, 4);
for b = 1:max(bw(:))
    region = (bw == b);
    if max(max(im(region))) < h
        bw(bw == b) = 0;
    end
end
nec = uint8(255*uint8(bw>0.5));


function im = smoothing(im, smh_w)

ker = zeros(3, 3);
for i = 1:3
    for j = 1:3
        ker(i, j) = 1.0 / (abs(i-2) + abs(j-2) + smh_w);
    end
end
ker = ker / sum(ker(:));
im = conv2(double(im), ker, 'same');

