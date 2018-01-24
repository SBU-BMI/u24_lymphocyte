function get_tumor_pos_neg_map(svs_name, username, width, height, mark_file)

PosLabel = 'VesselNormal';
NegLabel = 'NoneExist';
HeatPixelSize = 4;
PatchHeatPixelN = 25;
PatchSampleRate = 50;

tumor = get_labeled_im_high_res(mark_file, width, height, HeatPixelSize, PosLabel, NegLabel);
image_path = sprintf('tumor_heatmaps/%s.%s.png', svs_name, username);
imwrite(tumor', image_path);
get_tumor_region_extract(svs_name, username, image_path, width, height, PatchHeatPixelN, PatchSampleRate);

