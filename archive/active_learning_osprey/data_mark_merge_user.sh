#!/bin/bash

mkdir -p data_user_merged

for files in data/TCGA-*.txt; do
    echo $files | awk -F'data/|_' '{print $2}'
done | sort -u > /tmp/data_mark_merge_user_file_list.txt

while read svs; do
    user=`ls -l data/${svs}_* | sort -k 5 -nr | head -n 1 | awk -F'_' '{print $2}'`
    cp data/${svs}_${user}_*.txt data_user_merged/
done < /tmp/data_mark_merge_user_file_list.txt

exit 0
