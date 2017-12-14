function extract_feature_folder()

% Folder of imgs
img_dir = '/data09/shared/lehhou/theano/nucleus/nucleus_encoder/visual';
saved_file = '/data08/shared/lehhou/nucleus_encoder/matconvnet/feat_files/visual_vgg.mat';
is_parallel = true;
n_worker = 20;

temp = dir(fullfile(img_dir, '*.png'));
n_img = length(temp);

img_list = cell(n_img, 1);
for i_img = 1:n_img
    img_list{i_img} = ['case_' num2str(i_img) '.png'];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can copy this file to any directory
% Setup. Runing it only once
run /data05/shared/lehhou/matconvnet/matconvnet-1.0-beta20/matlab/vl_setupnn
net = load('/data05/shared/lehhou/matconvnet/imagenet-vgg-verydeep-16.mat');
net = vl_simplenn_tidy(net);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load matlab's image peppers.png and extract features

if (is_parallel == true)
    poolobj = parpool(n_worker);
end

vgg_feat = zeros(length(img_list), 4096);
parfor i_img = 1:length(img_list)
    img = single(imread(fullfile(img_dir, img_list{i_img})));
    vgg_feat(i_img,:) = vgg_feature(img, net);
    disp(i_img);
end

if (is_parallel == true)
    delete(poolobj);
end

save(saved_file, 'vgg_feat');

%img = single(imread('peppers.png'));
% fea is a row vector of 1x4096
%fea = vgg_feature(img, net);

end


function fea = vgg_feature(im_, net)

im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);

res = vl_simplenn(net, im_);

fea = squeeze(gather(res(end-2).x));
fea = fea';


end
