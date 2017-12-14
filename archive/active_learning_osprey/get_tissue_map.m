function t = get_tissue_map(wht)

h = fspecial('gaussian', 10, 2);
wht = conv2(wht, h, 'same');
t = uint8(wht > 12);

