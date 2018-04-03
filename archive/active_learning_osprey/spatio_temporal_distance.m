function res = spatio_temporal_distance(pt1, pt2, out_threshold_str, min_dist_str, min_time_str)

out_threshold = str2num(out_threshold_str);
min_dist = str2num(min_dist_str);
min_time = str2num(min_time_str);


below_threshold = 0.5*out_threshold;
above_threshold = 1.5*out_threshold;

pt1 = repmat(pt1, [size(pt2, 1), 1]);

% Time difference
time_diff = abs(pt1(:,3) - pt2(:,3));

% Spatio difference
dist_diff = pt1(:,1:2) - pt2(:,1:2);
dist_diff = dist_diff .^ 2;
dist_diff = sum(dist_diff, 2);
dist_diff = sqrt(dist_diff);

time_idx = (time_diff <= min_time);
dist_idx = (dist_diff <= min_dist);

small_dist_idx = (time_idx | dist_idx);

res = above_threshold * ones(size(pt1, 1), 1);
res(small_dist_idx) = below_threshold;


end
