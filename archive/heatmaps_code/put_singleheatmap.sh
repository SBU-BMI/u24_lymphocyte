#!/bin/bash

FILE=${1}
TRIMMED_FILE=${FILE:11}
HEATMAP="heatmap_${TRIMMED_FILE}.json"
META="meta_${TRIMMED_FILE}.json"


python gen_json.py ${FILE} 


mongoimport --port 27015 --host nfs011 -d u24_segmentation -c results ${HEATMAP}
mongoimport --port 27015 --host nfs011 -d u24_segmentation -c metadata ${META}
mongoimport --port 27015 --host nfs011 -d tcga_segmentation -c metadata ${META}
echo "------- DONE ${FILE} -------------------"


