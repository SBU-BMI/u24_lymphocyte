#!/bin/bash

for files in svs_tiles/*/; do
    SVS=`echo ${files} | awk -F'/' '{print $2}'`
    if [ -f /data01/tcga_data/tumor/luad/${SVS} ]; then
        echo ${SVS}
    fi
done > heatmap_slide_list.txt

exit 0
