#!/bin/bash

sshpass -p levu2016 scp ./json/meta_* ./json/heatmap_* lehhou@osprey.bmi.stonybrook.edu:/home/lehhou/heatmap_pipeline/json_todo/

exit 0
