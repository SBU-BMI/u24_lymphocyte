function merge_images()

n=7;
merge = uint8(255*ones(110*n-10, 520, 3, 'uint8'));
for i = 1 : 447
    imag = imread(sprintf('imag_%d.png', i));
    mask = imresize(imread(sprintf('mask_%d.png', i)), [100, 100]);
    nucl = imread(sprintf('nucl_%d.png', i));
    glob = imread(sprintf('glob_%d.png', i));
    outp = imread(sprintf('outp_%d.png', i));

    ind = rem(i-1, n);
    merge(ind*110+1:ind*110+100,   1:100, :) = imag;
    merge(ind*110+1:ind*110+100, 106:205, :) = cat(3, mask, mask, mask);
    merge(ind*110+1:ind*110+100, 211:310, :) = nucl;
    merge(ind*110+1:ind*110+100, 316:415, :) = glob;
    merge(ind*110+1:ind*110+100, 421:520, :) = outp;
    if ind == (n-1)
        imwrite(merge, sprintf('merge_%d.png', uint8(i/n)));
        merge = uint8(255*ones(110*n-10, 520, 3, 'uint8'));
    end
end


