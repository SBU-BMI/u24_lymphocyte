function add_center_rectangle()
% adds a green rectangle in the image center, to indicate the patch
% the pathologist to label for.

% x and y point of the top-left corner of the rectangle
s = 200;
% side length of the rectangle
l = 100;

files = dir('images/*.png');
for file = files'
    fn = file.name;
    im = imread(['images/', fn]);
    im(s:s+l, s, 2) = 255;
    im(s, s:s+l, 2) = 255;
    im(s:s+l, s+l, 2) = 255;
    im(s+l, s:s+l, 2) = 255;
    imwrite(im, ['images/', fn]);
end

