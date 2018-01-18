function picked_cluster = clusterPoints_spatio_temporal(pt_cloud, min_dist, min_time)

out_threshold = 1;

Z = linkage(pt_cloud, 'single', {'@spatio_temporal_distance', num2str(out_threshold), num2str(min_dist), num2str(min_time)});
clusters = mat2cell(pt_cloud, ones(size(pt_cloud,1), 1), size(pt_cloud,2));
picked_cluster_idx = ones(size(pt_cloud,1) + size(Z,1), 1);
for i = 1:size(Z, 1)
    if (Z(i,3) > out_threshold)
        break;
    end
    cl_idx = size(pt_cloud,1) + i;
    ele1 = Z(i,1);
    ele2 = Z(i,2);
    clusters(cl_idx) = {[clusters{ele1}; clusters{ele2}]};
    picked_cluster_idx(ele1) = 0;
    picked_cluster_idx(ele2) = 0;
end
 
picked_cluster_idx = picked_cluster_idx(1:size(clusters,1), :);
picked_cluster_raw = clusters(picked_cluster_idx == 1);

picked_cluster = cell(size(picked_cluster_raw));
for i_cluster = 1:size(picked_cluster_raw, 1)
    picked_cluster{i_cluster} = picked_cluster_raw{i_cluster}(:,1:2);
end

end

