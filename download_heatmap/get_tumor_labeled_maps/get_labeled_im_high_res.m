function tumor_im = get_labeled_im_high_res(mark_file, tot_width, tot_height, patch_size, PosLabel, NegLabel)

im_width = round(tot_width/patch_size);
im_height = round(tot_height/patch_size);

tumor_countor = {};
tumor_lab = {};
time_stamp = '0';
cur_c = [];
cur_lab = 0;

[~, ~, m_type, m_width, tses, poly] = textread(mark_file, '%s%s%s%d%s%s', 'delimiter', '\t', 'bufsize', 65536);

for iter = 1:length(m_type)
    mt = m_type{iter};
    ts = tses{iter};
    po = str2num(poly{iter});

    if (strcmp(mt, PosLabel))
        if isempty(cur_c) || ...
           (cur_lab == 1 && ...
            abs(str2num(time_stamp) - str2num(ts)) < 10 && ...
            (...
            ((cur_c(end-1)-po(1))^2 + (cur_c(end)-po(2))^2)^0.5 < 0.0001 || ...
            ((cur_c(1)-cur_c(end-1))^2 + (cur_c(2)-cur_c(end))^2)^0.5 > 0.0001 ...
            )...
           )
            cur_c = [cur_c, po];
        else
            tumor_countor{end+1} = cur_c;
            tumor_lab{end+1} = cur_lab;
            cur_c = po;
        end
        cur_lab = 1;
        time_stamp = ts;
    elseif (strcmp(mt, NegLabel))
        if isempty(cur_c) || ...
           (cur_lab == 0 && ...
            abs(str2num(time_stamp) - str2num(ts)) < 10 && ...
            (...
            ((cur_c(end-1)-po(1))^2 + (cur_c(end)-po(2))^2)^0.5 < 0.0001 || ...
            ((cur_c(1)-cur_c(end-1))^2 + (cur_c(2)-cur_c(end))^2)^0.5 > 0.0001 ...
            )...
           )
            cur_c = [cur_c, po];
        else
            tumor_countor{end+1} = cur_c;
            tumor_lab{end+1} = cur_lab;
            cur_c = po;
        end
        cur_lab = 0;
        time_stamp = ts;
    end
end

if ~isempty(cur_c)
    tumor_countor{end+1} = cur_c;
    tumor_lab{end+1} = cur_lab;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tumor_im = ones(im_width, im_height, 'uint8') * 127;
for i = 1:length(tumor_countor)
    tumor_im = tumor_poly_to_mask(tumor_im, tumor_countor{i}, tumor_lab{i});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, y] = defuck(x, y);
if length(x) < 50
    return;
end
x1 = x(5); y1 = y(5); x2 = x(7); y2 = y(7);
dup = 1;
i = 20;
while i < length(x)-6
    if ((abs(x(i)-x1) < 1e-4) && (abs(x(i+4)-x2) < 1e-4) && (abs(y(i)-y1) < 1e-4) && (abs(y(i+4)-y2) < 1e-4))
        dup = dup + 1;
        i = i + 40;
    end
    i = i + 1;
end
if dup > 1
    fprintf('dedup %d\n', dup);
    cut_len = round(length(x)/dup);
    x = x(1:cut_len);
    y = y(1:cut_len);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function im = tumor_poly_to_mask(im, countor, lab)
% poly_x, poly_y are inverted for poly2mask
poly_x = countor(1:2:end);
poly_y = countor(2:2:end);
%[poly_x, poly_y] = defuck(poly_x, poly_y);

fprintf('Tumor poly length %d [%d]\n', length(poly_x), lab);

if length(poly_x) > 1500
    poly_x = poly_x(1:round(length(poly_x)/500):end);
    poly_y = poly_y(1:round(length(poly_y)/500):end);
end

[poly_x, poly_y] = points2contour(poly_x, poly_y, 1, 'cw');
bw = poly2mask(size(im,2)*poly_y, size(im,1)*poly_x, size(im,1), size(im,2));
if lab == 1
    im = im  + 255 * uint8(bw>0.5);
else
    im = im .* uint8(bw<0.5);
end

