function picked_cluster = clusterPoints(pt_cloud, min_dist)
 
Z = linkage(pt_cloud, 'single');
clusters = mat2cell(pt_cloud, ones(size(pt_cloud,1), 1), size(pt_cloud,2));
picked_cluster_idx = ones(size(pt_cloud,1) + size(Z,1), 1);
for i = 1:size(Z, 1)
    if (Z(i,3) > min_dist)
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
picked_cluster = clusters(picked_cluster_idx == 1);
 
