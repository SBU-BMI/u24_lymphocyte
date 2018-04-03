#!/bin/bash

for f in prediction-TCGA-38-4631-01Z-00-DX1.5e0c873a-9c4c-4e0b-bf2e-e3cd8b760761; do
    #echo nohup python gen_json.py $f & 
    nohup bash put_singleheatmap.sh $f &
done

wait;

echo DONE;
