#!/bin/bash

for files in ./json/{meta,heatmap}_*; do
    bash ../util/scp_to_remote_camic.sh ${files} /home/lehhou/heatmap_pipeline/json_todo/
    sleep 2
done

exit 0
