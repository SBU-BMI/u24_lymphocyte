#!/bin/bash

source ../../conf/variables.sh

# This script download the human markups

# Specify which cancertype, username, and the list
# of slides' heatmap you want to download.
# For example:
#     ctype=brca
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
findapi_host=${MONGODB_HOST}
bash download_markings_weights.sh ${ctype} ${username} ${case_list} ${raw_marking_output_path} ${findapi_host}

# At the end, ${RAW_MARKINGS_PATH}/ folder should contain both
# the weight files and the markup files.
# For example:
#     ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1__x__rajarsi.gupta_mark.txt
#     ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1__x__rajarsi.gupta_weight.txt
# The *_mark.txt file has the following format:
# SlideID  UserName  MarkLabel  MarkWidth  TimeStamp  Coordinates(alternating<x,y>)

exit 0
