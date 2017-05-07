for i = 0:99
    im_name = ['image_' num2str(i) '.png'];
    gt_name = ['gt_' num2str(i) '.png'];
    pr_name = ['pred_' num2str(i) '.png'];
    figure(1); imshow(imread(im_name)); figure(2); imshow(imread(gt_name)); figure(3); imshow(imread(pr_name)>170);
    pause;
end