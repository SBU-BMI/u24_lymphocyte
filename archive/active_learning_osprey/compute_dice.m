function compute_dice(svs, u1, u2, f1, f2)
im1 = imread(f1);
im2 = imread(f2);

lym1 = uint8(im1(:,:,1)>1);
lym2 = uint8(im2(:,:,1)>1);

union = 2 * sum(sum(lym1 .* lym2));
size1 = sum(lym1(:));
size2 = sum(lym2(:));

fid = fopen('dices/dice.txt', 'a');
fprintf(fid, '%s\t%s\t%s\t%d\t%d\t%d\n', svs, u1, u2, union, size1, size2);
fclose(fid);

