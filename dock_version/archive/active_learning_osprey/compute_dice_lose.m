function compute_dice_lose(svs, u1, u2, f1, f2)
im1 = imread(f1);
im2 = imread(f2);

lym1 = uint8(im1(:,:,1)>1);
lym2 = uint8(im2(:,:,1)>1);

union1 = lose_union(lym1, lym2);
union2 = lose_union(lym2, lym1);
union = sum(sum(uint8(union1>=1))) + sum(sum(uint8(union2>=1)));
size1 = sum(lym1(:));
size2 = sum(lym2(:));

fid = fopen('dices/dice_lose.txt', 'a');
fprintf(fid, '%s\t%s\t%s\t%d\t%d\t%d\n', svs, u1, u2, union, size1, size2);
fclose(fid);

function union = lose_union(lym1, lym2)
union = lym1 .* lym2;
union(1:end-1, :) = union(1:end-1, :) + lym1(1:end-1, :) .* lym2(2:end  , :);
union(2:end  , :) = union(2:end  , :) + lym1(2:end  , :) .* lym2(1:end-1, :);
union(:, 1:end-1) = union(:, 1:end-1) + lym1(:, 1:end-1) .* lym2(:, 2:end  );
union(:, 2:end  ) = union(:, 2:end  ) + lym1(:, 2:end  ) .* lym2(:, 1:end-1);

