#!/bin/bash

source ../../conf/variables.sh

# This script download the human markups

# Specify which cancertype, username, and the list
# of slides' heatmap you want to download.
ctype=${CANCER_TYPE}
username=${USERNAME}
case_list=${CASE_LIST}
raw_marking_output_path=${RAW_MARKINGS_PATH}
bash download_markings.sh ${ctype} ${username} ${case_list} ${raw_marking_output_path}

# To fully reconstruct the human modified heatmap,
# you also need the human chosen lymphocyte sensitivity,
# necrosis specificity, and smoothness.
# You need to get those numbers manually from caMicroscope and put
# them in a file named ./data/${slide_id}_${username}.txt
# For example:
#     ./data/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt
# There should be three lines in the weight file, the first
# line shows the chosen lymphocyte sensitivity, the second
# line shows the chosen necrosis specificity, the last
# line shows the smoothness.
# For example:
#     cat ./data/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt
#     0.77
#     0.60
#     0.15

# At the end, ./data/ folder should contain both
# the weight files and the markup files.
# For example:
#    ./data/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_mark.txt
#    ./data/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt

exit 0
