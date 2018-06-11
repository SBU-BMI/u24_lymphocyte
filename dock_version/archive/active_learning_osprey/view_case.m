function view_case(id)

auto = imread(['rates/rate.', id, '.automatic.png']);
zhao = imread(['rates/rate.', id, '.azhao83.png']);
john = imread(['rates/rate.', id, '.john.vanarnam.png']);

f = figure;
movegui(f, 'center');
subplot(2, 2, 1), imshow(auto), title('auto');
subplot(2, 2, 2), imshow(zhao), title('zhao');
subplot(2, 2, 4), imshow(john), title('john');

