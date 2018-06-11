#!/bin/bash

while read line; do
    rm gened/prediction-${line}.*
    rm patch-level/prediction-${line}.*
done < clear_heatmap_files.txt

exit 0
