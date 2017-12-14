h5file = 'nuclei_image/test/TCGA-02-0440-01Z-00-DX1.4fef88c9-eff7-4e00-be19-d0db2871329a_appMag_20_8_6-seg.h5';
info = h5info(h5file);
image_num = info.Datasets(1).Dataspace.Size(2);
recon = uint8(load('recon.txt'));

for i = 1 : 8 * 6
    ind = ceil(rand() * image_num);
    im = h5read(h5file, '/data', [10 10 1 ind], [32, 32, 4, 1]);
    im = uint8(255 * im(:, :, 1 : 3));
    subplot(8, 6, i), imshow([im, zeros(32, 1, 3, 'uint8'), ...
                                reshape(recon(ind, :), 32, 32, 3), zeros(32, 1, 3, 'uint8'), ...
                                imresize(imresize(im, [7 7]), [32 32])]);
end

