#!/bin/bash

THUMB_DIR=/data01/shared/lehhou/thumbnails_extraction/
TUMOR_DIR=/data08/shared/hanle/tumor.tumor_region_only/data/tumor_labeled_heatmaps/

cat list.txt | while read SVS_ID; do
    THUMB=/data01/shared/lehhou/thumbnails_extraction/${SVS_ID}.png
    TUMOR=`ls /data08/shared/hanle/tumor.tumor_region_only/data/tumor_labeled_heatmaps/${SVS_ID}.*.png | head -n 1`
    if [ -f ${THUMB} ]; then
        if [ ! -z ${TUMOR} ]; then
            echo ${THUMB} ${TUMOR} ${SVS_ID}.png
            python add_tumor_contour.py ${THUMB} ${TUMOR} ${SVS_ID}.png
            if [ $? -ne 0 ]; then
                cp ${THUMB} ${SVS_ID}.png
            fi
        else
            cp ${THUMB} ${SVS_ID}.png
        fi
    else
        cp ${THUMB} ${SVS_ID}.png
    fi
done

exit 0
