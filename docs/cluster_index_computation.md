# Cluster Index Computation

## Prerequisites

To start these steps there are several prerequisites.  These instructions assume you have installed the base files under the folder "u24_lymphocyte" and all paths given here will be under that.  You must first run the TIL prediction phase and have prediction files in the data/heatmap_txt folder.  The associated images matching the prediction data you have need to be in "data/svs".

## Steps

Where noted, files are part of the Develop branch and need to copied from there.

1) Run "u24_lymphocyte/scripts/stratified_run_all.sh". The code is on Develop branch. In addition copy the following files from the Develop branch:
    * Use file patch_sampling/get_sample_list.m  from “Develop” branch
    * Use file patch_extraction_from_list/start.sh  from “Develop” branch
    Note: output from run will be N*10 patches in data/patches_from_heatmap where N is number of slides you are processing.
2) After the run is completed copy the patches to the web server setup for patch-labeling.  Code is on [github here](https://github.com/SBU-BMI/u24_lymphocyte/tree/develop/patch_labeling_web).  Have experts classify the patches into groups.
3) .
