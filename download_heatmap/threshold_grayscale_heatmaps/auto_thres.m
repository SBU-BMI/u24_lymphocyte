function auto_thres(in_folder, out_folder, threshold_list)

[svss, lymt, nect] = textread(threshold_list, '%s%f%f', 'delimiter', ' ');
lymcont = containers.Map(svss, lymt);
neccont = containers.Map(svss, nect);

fid = fopen([out_folder, '/infiltration_rate.csv'], 'w');
fprintf(fid, 'Slides,TIL\n');

files = dir([in_folder, '/*.png']);
for f = files'
    fname = f.name;
    save_fn = fname;

    svs_name = strsplit(fname, '.png');
    svs_name = svs_name{1};

    if isKey(lymcont, svs_name) < 0.5
        lt = 0.775;
    else
        lt = values(lymcont, {svs_name});
        lt = lt{1};
    end
    if isKey(neccont, svs_name) < 0.5
        nt = 0.775;
    else
        nt = values(neccont, {svs_name});
        nt = nt{1};
    end
    fprintf('thresholds: %s %f %f\n', svs_name, lt, nt);

    im = imread([in_folder, '/', fname]);
    lym = lym_magic_thres(im(:, :, 1), lt);
    nec = nec_magic_thres(im(:, :, 2), nt);
    lym = lym .* uint8(nec < 1);

    im(:, :, 1) = lym .* uint8(im(:, :, 3) > 1);
    im(:, :, 3) = im(:, :, 3) .* uint8(lym<0.5);
    im(:, :, 2) = 0;
    
    % ===== change Red color to white
    %t = im(:,:,3);
    %t(im(:,:,1) > 0) = 255; im(:,:,3) = t;
    %t = im(:,:,2);
    %t(im(:,:,1) > 0) = 255; im(:,:,2) = t;
    % ==== end changing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nlym = sum(sum(double(im(:, :, 1)>0.5)));
    ntissue = nlym + sum(sum(double(im(:, :, 3)>0.5)));
    fprintf(fid, '%s,%.6f\n', svs_name, nlym/ntissue);
    
    im(:, :, 2) = im(:, :, 1);
    im(:, :, 1) = 0;
    imwrite(im, [out_folder, '/', save_fn]);

end
fclose(fid);


function lym = lym_magic_thres(im, lt)

h = (1.0-lt)*255.0;
l = (1.0-lt)*255.0;
w = 0.15;
w = 0.15; % Han changed 6/28/18

im = smoothing(im, 0.01+2*w*w);
bw = bwlabel(im>=l, 4);
for b = 1:max(bw(:))
    region = (bw == b);
    if max(max(im(region))) < h
        bw(bw == b) = 0;
    end
end
lym = uint8(255*uint8(bw>0.5));


function nec = nec_magic_thres(im, nt)

h = nt*255.0;
l = nt*255.0;
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

