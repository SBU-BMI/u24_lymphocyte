function get_tumor_region_extract(svs_name, username, image_path, width, height, patch_heat_pixel_n, sample_rate)
% patch_heat_pixel_n = 25;	% 25 --> patch size of 100.
% sample_rate = 50;
% image_path = path to the heatmap file which is scale down (4 times) the size of the slide
% the index in the name of the patches are position of the patch on the HEATMAP

% generate the txt files in folder tumor_image_to_extract/

labs = floor(patch_heat_pixel_n/2-0.5);
labe = floor(patch_heat_pixel_n/2);

im = imread(image_path);
im_height = size(im, 1);
im_width = size(im, 2);

[ys, xs] = find(im==0 | im==255);
rp = randperm(length(xs));
xs = xs(rp(1:sample_rate:end));	% randomly sample the patches so they are not close to each other
ys = ys(rp(1:sample_rate:end));

fid = fopen(['tumor_image_to_extract/', svs_name, '.', username, '.txt'], 'w');
count_neg = 0;	% count number of negative patch
count_pos = 0;
heatmap_ratio = width/im_width;	% default setting is 4
lab_width = 100/heatmap_ratio; 	% default is 25
labs_center = floor(lab_width/2-0.5);
labe_center = floor(lab_width/2);

for i = 1:length(xs)
    if (xs(i)-labs < 1 || ys(i)-labs < 1 || xs(i)+labe > im_width || ys(i)+labe > im_height)
        continue;
    end
    lab_patch = im(ys(i)-labs:ys(i)+labe, xs(i)-labs:xs(i)+labe);
    lab_center_100 = im(ys(i)-labs_center:ys(i)+labe_center, xs(i)-labs_center:xs(i)+labe_center);

    patch_name = sprintf('%s-%d-%d', svs_name, ...
        int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1)); 
   
    % generate patch label form the patch's heatmap
    threshold = 0.5;
    label = -1;
    label_0 = length(find(lab_center_100 == 0));
    label_1 = length(find(lab_center_100 == 255));
    if label_0/(lab_width*lab_width) > threshold
        label = 0;
        count_neg = count_neg + 1;
    elseif label_1/(lab_width*lab_width) > threshold
        label = 1;
        count_pos = count_pos + 1;
    end
    
    % only save the heatmaps and labels if they are Neg/Pos patches, unknown patches will not be saved
    if label >= 0
        % width/im_width: scaling factor between the slide's size and heatmap'size
        fprintf(fid, '%d,%d,%d,%d,%d\n', ...
            int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
            int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1), ...
            label);
        imwrite(lab_patch, sprintf('tumor_ground_truth/%s-%d-%d-%d-%d.png', svs_name, ...
            int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
            int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1)));
    end
end
fid_label_sum = fopen(['tumor_ground_truth/label_summary.txt'], 'a');
fprintf(fid_label_sum, '%s %d %d [svs, #Neg, #Pos]\n', svs_name, count_neg, count_pos);	% write number of Neg, Pos to the end of the file

fclose(fid);
fclose(fid_label_sum);
