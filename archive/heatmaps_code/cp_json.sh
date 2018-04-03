#!/bin/bash

sshpass -p levu2016 scp ../heatmaps_v3/meta_TCGA-* ../heatmaps_v3/heatmap_TCGA-* lehhou@osprey.bmi.stonybrook.edu:/home/lehhou/heatmap/
#scp ../heatmaps_v3/meta_TCGA-* ../heatmaps_v3/heatmap_TCGA-* lehou@129.49.249.191:/home/lehou/heatmap/

exit 0
