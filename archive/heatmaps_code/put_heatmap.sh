#!/bin/bash

for f in $(ls -1 prediction-TCGA* | awk '!/.txt/ && !/.csv/'); do
    #echo nohup python gen_json.py $f & 
    nohup bash put_singleheatmap.sh $f &
done

wait;

echo DONE;
