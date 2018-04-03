function get_tumor_region_extract(svs_name, username, image_path, width, height, patch_heat_pixel_n, sample_rate)

labs = floor(patch_heat_pixel_n/2-0.5);
labe = floor(patch_heat_pixel_n/2);

im = imread(image_path);
im_height = size(im, 1);
im_width = size(im, 2);

[ys, xs] = find(im==0 | im==255);
rp = randperm(length(xs));
xs = xs(rp(1:sample_rate:end));
ys = ys(rp(1:sample_rate:end));

fid = fopen(['tumor_image_to_extract/', svs_name, '.', username, '.txt'], 'w');
for i = 1:length(xs)
    if (xs(i)-labs < 1 || ys(i)-labs < 1 || xs(i)+labe > im_width || ys(i)+labe > im_height)
        continue;
    end
    lab_patch = im(xs(i)-labs:xs(i)+labe, ys(i)-labs:ys(i)+labe);

    fprintf(fid, '%d,%d,%d,%d,%d\n', ...
        int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
        int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1), ...
        -1);
    imwrite(lab_patch, sprintf('tumor_ground_truth/%s-%d-%d-%d-%d.png', svs_name, ...
        int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
        int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1)));
end
fclose(fid);

