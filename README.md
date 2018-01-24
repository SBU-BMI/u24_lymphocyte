# u24_lymphocyte

This software implements the pipeline for the lymphocyte classification project. 

List of folders and functionalities are below: 

scripts/: contains scripts that connect several sub-functionalities together for complete functionalities such as generating camicroscope heatmaps given svs images.

conf/: contains configuration. 

csv_generation/: generates csv files which are the input of the clustering algorithm. 

data/: a place where should contain all logs, input/output images, trained CNN models, and large files. 

download_heatmap/: downloads grayscale lymphocyte or tumor heatmaps, and thresholds grayscale heatmaps to binary heatmaps. 

heatmap_gen/: generate json files that represents heatmaps for camicroscope, using the lymphocyte and necrosis CNNs' raw output txt files. 

patch_extraction/: extracts all patches from svs images. Mainly used in the test phase. 

patch_extraction_from_list/: extracts patches from a list. Used in scenarios like extracting training patches, and extracting stratified sampled patches for rethresholding purpose. 

patch_labeling_web/: a website that label each image as positive/negative/ignore. 

patch_sampling/: samples patches in a stratified manner. Input: grayscale heaetmaps. Output: a list of patches for patch_extraction_from_list/ to extract. 

prediction/: CNN prediction code. 

training/: CNN training code. 

third_party/: third party code for CNN. 
