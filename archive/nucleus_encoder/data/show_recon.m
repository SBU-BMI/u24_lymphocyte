function show_recon()

r = uint8(load('./recon.txt'));
x = uint8(load('./X.txt'));
%r = permute(reshape(r, 81, 3, 50, 50), [4, 3, 2, 1]);
%x = permute(reshape(x, 81, 3, 50, 50), [4, 3, 2, 1]);
r = reshape(r, 50, 50, 3, []);
x = reshape(x, 50, 50, 3, []);
shuf = randperm(size(r, 4));
r = r(:, :, :, shuf);
x = x(:, :, :, shuf);

ind = 1;
for i = 1:9
    for j = 1:9
        ori = squeeze(x(:, :, :, ind));
        rec = squeeze(r(:, :, :, ind));
        subplot(9, 9, ind), imshow(cat(2, ori, rec));
        ind = ind + 1;
    end
end

