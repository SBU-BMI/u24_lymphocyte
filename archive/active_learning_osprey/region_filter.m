function [pred, necr, truth, tumor, whiteness] = region_filter(pred, necr, truth, tumor, whiteness, svs_name)

fname = ['region_for_badcases/', svs_name, '.txt'];
if exist(fname, 'file') > 0.5
    mat = load(fname);
    y1 = mat(1);
    y2 = mat(2);
    x1 = mat(3);
    x2 = mat(4);

    s1 = size(pred, 1);
    s2 = size(pred, 2);
    filt = zeros(s1, s2, 'uint8');
    filt(ceil(s1*x1+0.1) : floor(s1*x2-0.1), ceil(s2*y1+0.1) : floor(s2*y2-0.1)) = 1;

    pred = pred .* double(filt);
    necr = necr .* double(filt);
    truth = truth .* filt;
    tumor = tumor .* filt;
    whiteness = whiteness .* double(filt);
end

