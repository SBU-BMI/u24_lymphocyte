#!/bin/bash

rm deep_conv_classification_alt36_heatmap.txt
for files in svs_tiles/TCGA-*; do
    if [ -f ${files}/patch-level-pred.txt ]; then
        echo ${files}/patch-level-pred.txt existed
    else
        echo ${files} | awk -F'/' '{print $NF}' >> deep_conv_classification_alt36_heatmap.txt
    fi
done

