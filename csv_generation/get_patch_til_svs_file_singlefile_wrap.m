% Eg
% cancer_type:    'luad'
% parent_out_dir (output directory): '/data07/shared/lehhou/lym_outputs/csv'
% src_dir (source directory):        '/data08/shared/lehhou/active_learning_osprey'
% slide_id (org name/id of the slide): 'TCGA-FD-A6TK-01Z-00-DX1'

function get_patch_til_svs_file_singlefile_wrap(cancer_type, parent_out_dir, src_dir, slide_id, cancertype_path)


%parent_out_dir = '/data07/shared/lehhou/lym_outputs/csv';
out_dir = fullfile(parent_out_dir, cancer_type);
if (exist(out_dir) == 0)
    mkdir(out_dir);
end


%src_img_dir = [src_dir '/rates-' cancer_type '-all-auto'];
src_img_dir = cancertype_path;
bin_src_imgs = dir(fullfile(src_img_dir, ['rate.' slide_id '.automatic.png']));

%parpool(12);

for i_img = 1:length(bin_src_imgs)
    slide_name = bin_src_imgs(i_img).name(6:end-14);
    csv_path = fullfile(out_dir, [slide_name '.csv']);
    info_path = fullfile(out_dir, [slide_name '.info']);
    fileID = fopen(csv_path, 'w');
    infoID = fopen(info_path, 'w');
    
    bin_img_name = [bin_src_imgs(i_img).name(1:end-4) '_thres.png'];
    real_img_name = bin_src_imgs(i_img).name;
    real_img_path = fullfile(src_img_dir, real_img_name);
    bin_img_path = fullfile(src_img_dir, bin_img_name);
    
    real_img = imread(real_img_path);
    bin_img = imread(bin_img_path);
    
    width = size(real_img, 2);
    height = size(real_img, 1);

    n_tissue = 0;
    n_til = 0;
    
    for iH = 1:height
        for iW = 1:width
            % real value
            real_value = double(real_img(iH, iW, 1)) / 255.0;
            
            % bin value
            bin_value = 0;
            % if this is pos tile
            if (bin_img(iH, iW, 1) > bin_img(iH, iW, 3))
                bin_value = 1;
                n_til = n_til + 1;
                n_tissue = n_tissue + 1;
            end

            % if this is a tissue tile
            if (bin_img(iH, iW, 3) > 128)
                n_tissue = n_tissue + 1;
            end
            
            fprintf(fileID, '%d,%d,%d,%.4f\n', iH, iW, bin_value, real_value);
        end
    end

    fprintf(infoID, '%s,%d\n', 'tissue_number', n_tissue);
    fprintf(infoID, '%s,%d\n', 'til_number', n_til);
    
    fclose(fileID);
    fclose(infoID);
end

%delete(gcp);
end

