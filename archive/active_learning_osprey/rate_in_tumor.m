function rate_in_tumor(pred_folder, tumor_folder, final_folder)

fid = fopen([final_folder, '/infiltration_rate.txt'], 'w');
fprintf(fid, 'Slides,TIL\n');

files = dir([tumor_folder, '/*.png']);
for f = files'
    fname = f.name;
    tumor = imread([tumor_folder, '/', fname]);
    sp = strsplit(fname, '.');

    caseid = sp{2};
    im = imread([pred_folder, '/', caseid, '.png']);
    im(:, :, 2) = tumor(:, :, 2);

    imwrite(im, [final_folder, '/', caseid, '.png']);
    nlym = sum(sum(double(im(:, :, 1)>0.5) .* double(im(:, :, 2)>0.5)));
    ntissue = nlym + sum(sum(double(im(:, :, 3)>0.5) .* double(im(:, :, 2)>0.5)));
    fprintf(fid, '%s,%.6f\n', caseid, nlym/ntissue);
end

fclose(fid);

