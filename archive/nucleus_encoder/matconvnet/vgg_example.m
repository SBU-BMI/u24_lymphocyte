function vgg_example()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You can copy this file to any directory
% Setup. Runing it only once
run /data05/shared/lehhou/matconvnet/matconvnet-1.0-beta20/matlab/vl_setupnn
net = load('/data05/shared/lehhou/matconvnet/imagenet-vgg-verydeep-16.mat');
net = vl_simplenn_tidy(net);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load matlab's image peppers.png and extract features
img = single(imread('peppers.png'));
% fea is a row vector of 1x4096
fea = vgg_feature(img, net);




function fea = vgg_feature(im_, net)

im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage);

res = vl_simplenn(net, im_);

fea = squeeze(gather(res(end-2).x));
fea = fea';

