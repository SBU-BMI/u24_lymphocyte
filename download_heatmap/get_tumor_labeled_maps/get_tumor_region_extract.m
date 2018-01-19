function get_tumor_region_extract(svs_name, username, image_path, width, height)

labs = 5;
labe = 6;
siz = labe - labs + 1;

im = transpose(imread(image_path));

[xs, ys] = find(im==0 | im==255);
rp = randperm(length(xs));
xs = xs(rp(1:17:end));
ys = ys(rp(1:17:end));

fid = fopen(['tumor_image_to_extract/', svs_name, '.', username, '.txt'], 'w');
for i = 1:length(xs)
    if (xs(i)-labs < 1 || ys(i)-labs < 1 || xs(i)+labe > size(im,1) || ys(i)+labe > size(im,2))
        continue;
    end
    lab_patch = im(xs(i)-labs:xs(i)+labe, ys(i)-labs:ys(i)+labe);

    imwrite(lab_patch', sprintf('tumor_ground_truth/lab_%s_%s_%.8f_%.8f.png', ...
        svs_name, username, (xs(i)-0.5)/size(im,1), (ys(i)-0.5)/size(im,2)));

    fprintf(fid, '%d,%d,%d,%d\n', ...
        int64((xs(i)-labs-1)/size(im,1)*width+1), int64((ys(i)-labs-1)/size(im,2)*height+1), ...
        int64((siz/size(im,1)*width + siz/size(im,2)*height) / 2.0), -1);
end
fclose(fid);

