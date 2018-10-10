#!/bin/bash

CLUSTER_FOLDER=/data01/shared/hanle/SEER_clustering/output/cluster_indices_whole_slides/
HE_FOLDER=/data01/shared/lehhou/thumbnails_tumor_contour/
TIL_FOLDER=/data01/shared/lehhou/SEER-VTR/389-all/u24_lymphocyte/data/thresholded_heatmaps/

N=1
cat list.txt | while read SVS_ID; do
    HE=${HE_FOLDER}/${SVS_ID}.png
    CLUSTER=${CLUSTER_FOLDER}/${SVS_ID}_clusters_ap.jpg
    TIL=${TIL_FOLDER}/${SVS_ID}.png
    if [ -f $HE ] && [ -f $TIL ]; then
        cp $HE images/${N}_HE.png
        if [ -f $CLUSTER ]; then
            python transpose.py $CLUSTER images/${N}_cluster.jpg
        fi
        cp $TIL images/${N}_TIL.png
        echo ${SVS_ID} > images/${N}_info.txt
        N=$((N+1))
    else
        echo $HE $CLUSTER $TIL
    fi
done

exit 0
