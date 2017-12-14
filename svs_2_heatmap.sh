#!/bin/bash

cd patch_extraction
nohup bash start.sh &> log/log.patch_extraction_start.txt &
cd ..

cd prediction
nohup bash start.sh &> log/log.prediction_start.txt &
cd ..

wait;

cd heatmap_gen
nohup bash start.sh &> log/log.heatmap_gen_start.sh &
cd ..

wait;

exit 0
