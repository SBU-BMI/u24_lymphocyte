function get_tumor_region_extract(svs_name, username, image_path, width, height, patch_heat_pixel_n, sample_rate)
% patch_heat_pixel_n = 25;	% how to use this param??? 25 --> patch size of 100.
% sample_rate = 50;
% image_path = path to the heatmap file which is scale down (4 times) the size of the slide
% the index in the name of the patches are position of the patch on the HEATMAP

% generate the txt files in folder tumor_image_to_extract/

labs = floor(patch_heat_pixel_n/2-0.5);
labe = floor(patch_heat_pixel_n/2);

im = imread(image_path);
im_height = size(im, 1);
im_width = size(im, 2);

[ys_1, xs_1] = find(im==255); % for label 1
rp = randperm(length(xs_1));
xs_1 = xs_1(rp(1:sample_rate:end));	% randomly sample the patches so they are not close to each other
ys_1 = ys_1(rp(1:sample_rate:end));

[ys_0, xs_0] = find(im == 0); % for label 0
rp = randperm(length(xs_0));
sample_rate_0 = sample_rate*1;
%neg_pos_ratio = 3; sample_rate_0 = max(floor(length(xs_0)/length(xs_1)/neg_pos_ratio), sample_rate);
xs_0 = xs_0(rp(1:sample_rate_0:end));
ys_0 = ys_0(rp(1:sample_rate_0:end));

fid = fopen(['tumor_image_to_extract/', svs_name, '.', username, '.txt'], 'w');
%%% ========= extract label 0 and 1 first =======
xs = [xs_1; xs_0];  % xs is col vector
ys = [ys_1; ys_0];
count_neg = 0; count_pos = 0; count_bg = 0; count_g = 0; count_y = 0;

xy_good = zeros(2, length(xs) + 1);
count_good = 1;
for i = 1:length(xs)
    if (xs(i)-labs < 1 || ys(i)-labs < 1 || xs(i)+labe > im_width || ys(i)+labe > im_height)
        continue;
    end
    %fprintf('%d %d %d %d %d %d\n', size(im, 1), size(im, 2), labs, labe, xs(i), ys(i));
    lab_patch = im(ys(i)-labs:ys(i)+labe, xs(i)-labs:xs(i)+labe);
    area = size(lab_patch, 1)*size(lab_patch, 2);

    % generate patch label form the patch's heatmap
    threshold = 0.5;
    label = -10;
    label_1 = length(find(lab_patch  == 255));
    if label_1/area > threshold && i <= length(xs_1)
        label = 1;
    end
    label_0 = length(find(lab_patch == 0));
    if label_0/area > threshold && i > length(xs_1)
        label = 0;
    end
    
    % only save the heatmaps and labels if they are Neg/Pos patches, unknown patches will not be saved
    if label > -10
        clear temp
        temp = repmat([xs(i); ys(i)],1,count_good);
        dist = sqrt(sum((temp - xy_good(:,1:count_good)).^2, 1));
        if ~isempty(find(dist < sample_rate, 1))
            label = -10;
        else
            count_good = count_good + 1;
            xy_good(:, count_good) = [xs(i); ys(i)];
        end
    end

    if label == 1
        count_pos = count_pos + 1;
    end
    if label == 0
        count_neg = count_neg + 1;
    end

    if label > -10
        % width/im_width: scaling factor between the slide's size and heatmap'size
        fprintf(fid, '%d,%d,%d,%d,%d\n', ...
            int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
            int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1), ...
            label);
        if i < 100      % only save first 100 patches to test
            imwrite(lab_patch, sprintf('tumor_ground_truth/%s-%d-%d-%d-%d.png', svs_name, ...
                int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
                int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1)));
        end
    end
end
%%% ======= end of extracting label 0 & 1 =========

No_label_0_1 = count_good;
clear xs ys xy_good;

function [ys_bg, xs_bg] = get_samples_other_class(color_value, sample_rate_base, neg_pos_ratio)
    [ys_bg, xs_bg] = find(im ==  color_value); % for label background
    if length(xs_bg) > 10
        rp = randperm(length(xs_bg));
        sample_rate_bg = max(floor(length(xs_bg)/No_label_0_1/neg_pos_ratio), sample_rate_base);
        xs_bg = xs_bg(rp(1:sample_rate_bg:end));
        ys_bg = ys_bg(rp(1:sample_rate_bg:end));
    else 
        ys_bg = []; xs_bg = [];
    end
end

[ys_bg, xs_bg] = get_samples_other_class(100, sample_rate, 3); % background label = -1
[ys_g, xs_g] = get_samples_other_class(150, sample_rate, 2); % green label = 2
[ys_y, xs_y] = get_samples_other_class(50, sample_rate, 2); % red labels = -2

xs = [xs_bg; xs_g; xs_y];  % xs is col vector
ys = [ys_bg; ys_g; ys_y];

% ignore green and yellow for now
xs = xs_bg; ys = ys_bg;

len_xs_bg = length(xs_bg); len_xs_g = length(xs_g);
groups = ones(1, length(xs));   % label background as group 1
start_i = len_xs_bg + 1; groups(start_i:end) = groups(start_i:end) + 1;    % label green as group 2
start_i = start_i + len_xs_g; groups(start_i:end) = groups(start_i:end) + 1;    % label yellow as group 3

xy_good = zeros(2, length(xs) + 1);
count_good = 1;

for i = 1:length(xs)
    if (xs(i)-labs < 1 || ys(i)-labs < 1 || xs(i)+labe > im_width || ys(i)+labe > im_height)
        continue;
    end
    %fprintf('%d %d %d %d %d %d\n', size(im, 1), size(im, 2), labs, labe, xs(i), ys(i));
    lab_patch = im(ys(i)-labs:ys(i)+labe, xs(i)-labs:xs(i)+labe);
    area = size(lab_patch, 1)*size(lab_patch, 2);

    % generate patch label form the patch's heatmap
    threshold = 0.5;
    label = -10;
    switch groups(i)
        case 1
            label_bg = length(find(lab_patch == 100));
            if label_bg/area > threshold
                label = -1;
            end        
        case 2
            label_g = length(find(lab_patch == 150));
            if label_g/area > threshold
                label = 2;
            end
        case 3
            label_y = length(find(lab_patch == 50));
            if label_y/area > threshold
                label = -2;
            end         
    end
    
    % only save the heatmaps and labels if they are Neg/Pos patches, unknown patches will not be saved
    if label > -10
        clear temp
        temp = repmat([xs(i); ys(i)],1,count_good);
        dist = sqrt(sum((temp - xy_good(:,1:count_good)).^2, 1));
        if ~isempty(find(dist < sample_rate, 1))
            label = -10;
        else
            count_good = count_good + 1;
            xy_good(:, count_good) = [xs(i); ys(i)];
        end
    end
    
    if label == -1
        count_bg = count_bg + 1;
    end
    if label == 2
        count_g = count_g + 1;
    end
    if label == -2
        count_y = count_y + 1;
    end

    if label > -10
        % width/im_width: scaling factor between the slide's size and heatmap'size
        fprintf(fid, '%d,%d,%d,%d,%d\n', ...
            int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
            int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1), ...
            label);
        if i < 100      % only save first 100 patches to test
            imwrite(lab_patch, sprintf('tumor_ground_truth/%s-%d-%d-%d-%d.png', svs_name, ...
                int64((xs(i)-labs-1)/im_width*width+1), int64((ys(i)-labs-1)/im_height*height+1), ...
                int64((xs(i)+labe  )/im_width*width+1), int64((ys(i)+labe  )/im_height*height+1)));
        end
    end
end

fid_label_sum = fopen(['tumor_ground_truth/label_summary.txt'], 'a');
fprintf(fid_label_sum, '%s %d %d %d %d %d [svs, #Neg, #Pos, #Background, #Green, #Yellow]\n', ...
            svs_name, count_neg, count_pos, count_bg, count_g, count_y);	% write number of Neg, Pos to the end of the file
l_1 = length(find(im==255));
l_0 = length(find(im==0));
l_bg = length(find(im==100));
l_g = length(find(im==150));
l_y = length(find(im==50));
fprintf(fid_label_sum, '%s %d %d %d %d %d [svs, #Neg_pixel, #Pos, #Background, #Green, #Yellow]\n', ...
            svs_name, l_0, l_1, l_bg, l_g, l_y);	% write number of Neg, Pos to the end of the file

fclose(fid);
fclose(fid_label_sum);
end
