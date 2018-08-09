% Eg
% cancer_type:    'luad'
% parent_out_dir (output directory): '/data07/shared/lehhou/lym_outputs/csv'
% src_dir (source directory):        '/data08/shared/lehhou/active_learning_osprey'
% group_width:    5
% group_height:   5
function get_patch_til_group_svs_file_wrap(cancer_type, parent_out_dir, src_dir, group_width, group_height)

group_width = 5;
group_height = 5;

%parent_out_dir = '/data07/shared/lehhou/lym_outputs/csv';
out_dir = fullfile(parent_out_dir, cancer_type);
if (exist(out_dir) == 0)
    mkdir(out_dir);
end


%src_img_dir = ['/data08/shared/lehhou/active_learning_osprey/rates-' cancer_type '-all-auto'];
src_img_dir = [src_dir '/rates-' cancer_type '-all-auto'];
bin_src_imgs = dir(fullfile(src_img_dir, '*automatic.png'));

parpool(12);

parfor i_img = 1:length(bin_src_imgs)
    slide_name = bin_src_imgs(i_img).name(6:end-14);
    csv_path = fullfile(out_dir, [slide_name '.csv']);
    fileID = fopen(csv_path, 'w');
    
    bin_img_name = [bin_src_imgs(i_img).name(1:end-4) '_thres.png'];
    real_img_name = bin_src_imgs(i_img).name;
    real_img_path = fullfile(src_img_dir, real_img_name);
    bin_img_path = fullfile(src_img_dir, bin_img_name);
    
    real_img = imread(real_img_path);
    bin_img = imread(bin_img_path);
    
    width = size(real_img, 2);
    height = size(real_img, 1);
    
    for iH = 1:group_height:height
        for iW = 1:group_width:width
            boundH = min(iH + group_height - 1, height);
            boundW = min(iW + group_width - 1, width);
            count = 0;
            sum_val = 0;
            for h_idx = iH:boundH
                for w_idx = iW:boundW
                    % real value
                    %real_value = double(real_img(h_idx, w_idx, 1)) / 255.0;

                    % bin value
                    bin_value = 0;
                    % if this is pos tile
                    if (bin_img(h_idx, w_idx, 1) > bin_img(h_idx, w_idx, 3))
                        bin_value = 1;
                    end
                    
                    sum_val = sum_val + bin_value;
                    count = count + 1;
                end
            end
            
            if (sum_val > 0 && sum_val < 25)
                a = 3;
            end
            ratio = double(sum_val) / double(count);
            x_center = int64((iW + boundW) / 2);
            y_center = int64((iH + boundH) / 2);
            fprintf(fileID, '%d,%d,%.4f\n', x_center, y_center, ratio);
        end
    end
    
    fclose(fileID);
end

delete(gcp);
end

