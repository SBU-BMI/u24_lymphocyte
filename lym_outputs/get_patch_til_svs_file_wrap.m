% Eg
% cancer_type:    'luad'
% parent_out_dir (output directory): '/data07/shared/lehhou/lym_outputs/csv'
% src_dir (source directory):        '/data08/shared/lehhou/active_learning_osprey'

function get_patch_til_svs_file(cancer_type, parent_out_dir, src_dir)


%parent_out_dir = '/data07/shared/lehhou/lym_outputs/csv';
out_dir = fullfile(parent_out_dir, cancer_type);
if (exist(out_dir) == 0)
    mkdir(out_dir);
end


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
    
    for iH = 1:height
        for iW = 1:width
            % real value
            real_value = double(real_img(iH, iW, 1)) / 255.0;
            
            % bin value
            bin_value = 0;
            % if this is pos tile
            if (bin_img(iH, iW, 1) > bin_img(iH, iW, 3))
                bin_value = 1;
            end
            
            fprintf(fileID, '%d,%d,%d,%.4f\n', iH, iW, bin_value, real_value);
        end
    end
    
    fclose(fileID);
end

delete(gcp);
end

