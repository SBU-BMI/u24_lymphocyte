function check_one_patch(txtf, imf)
a = load(txtf);

coor = ([round(a(:,1)/1984.01724137931/0.05-300), round(a(:,2)/1984.01724137931/0.05) - 680]+1)/2;
for i=1:100;
    im(coor(i, 2), coor(i, 1)) = a(i, 3);
end

imwrite(im, imf);

