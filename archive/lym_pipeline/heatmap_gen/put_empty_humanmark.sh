#!/bin/bash

ls -1 json/heatmap*.low_res.json | awk -F'heatmap_|.low_res.json' '{print $2}' | sort -u > /data08/shared/lehhou/camicro_code/unknown.txt
cd /data08/shared/lehhou/camicro_code/
bash gen_put_heatmap.sh unknown.txt unknown

exit 0
