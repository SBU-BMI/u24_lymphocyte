#!/bin/bash

source ../../conf/variables.sh

# This script download the human markups

# Specify which cancertype, username, and the list
# of slides' heatmap you want to download.
# For example:
#     ctype=BRCA
#     username=rajarsi.gupta@stonybrook.edu
#     case_list=/DATA/PATH/case_list.txt
# The content in case_list:
#     cat /DATA/PATH/case_list.txt
#     TCGA-05-4244-01Z-00-DX1
#     TCGA-05-4245-01Z-00-DX1
#     TCGA-05-4249-01Z-00-DX1
#     ...
ctype=${CANCER_TYPE}
username=${USERNAME}
case_list=${CASE_LIST}
raw_marking_output_path=${RAW_MARKINGS_PATH}
bash download_markings.sh ${ctype} ${username} ${case_list} ${raw_marking_output_path}

# To fully reconstruct the human modified heatmap,
# you also need the human chosen lymphocyte sensitivity,
# necrosis specificity, and smoothness.
# You need to get those numbers manually from caMicroscope and put
# them in a file named ${RAW_MARKINGS_PATH}/${slide_id}_${username}.txt
# For example:
#     ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt
# There should be three lines in the weight file, the first
# line shows the chosen lymphocyte sensitivity, the second
# line shows the chosen necrosis specificity, the last
# line shows the smoothness.
# For example:
#     cat ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt
#     0.77
#     0.60
#     0.15

# At the end, ${RAW_MARKINGS_PATH}/ folder should contain both
# the weight files and the markup files.
# For example:
#     ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_mark.txt
#     ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt
# The *_mark.txt file has the following format:
# SlideID  UserName  MarkLabel  MarkWidth  TimeStamp  Coordinates(alternating<x,y>)
# For example:
#     TCGA-3C-AALI-01Z-00-DX2 rajarsi.gupta@stonybrook.edu    TumorPos        1       1501535710324   0.1362388819418,0.97854705021941,0.136239,0.978547,0.135192,0.978547,0.134145,0.978547,0.134145,0.978547,0.132351,0.978547,0.130556,0.978547,0.128762,0.978547,0.126967,0.978547,0.125173,0.978547,0.123378,0.978547,0.121584,0.978547,0.121584,0.978547,0.119781,0.978222,0.117978,0.977897,0.116175,0.977572,0.114373,0.977247,0.11257,0.976922,0.110767,0.976597,

exit 0
