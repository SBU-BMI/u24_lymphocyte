function remove_necrosis(folder)

files = dir([folder, '/*.png']);
for f = files'
    fname = f.name;
    fprintf('%s\n', fname);

    im = imread([folder, '/', fname]);
    im(:, :, 2) = 0;

    imwrite(im, [folder, '/', fname]);
end

