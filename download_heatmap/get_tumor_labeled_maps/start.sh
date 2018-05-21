#!/bin/bash

source ../../conf/variables.sh

# This script takes input from ./download_markings/data/
# Check out ./download_markings/main.sh on how to prepare the input files

# Input folder to the downloaded markings
# This script will merge the markings
# with the unmodified heatmaps to output
# modified heatmaps.
# This script assumes that under this folder, there are
# two types of files:
#   1. files contain markings
#   2. files have selected lym&necrosis&smoothness weights
# For example:
#   ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_mark.txt
#   ${RAW_MARKINGS_PATH}/TCGA-NJ-A55O-01Z-00-DX1_rajarsi.gupta_weight.txt
# Checkout ./download_markings/main.sh on how to prepare those input files
MARKING_FOLDER=${RAW_MARKINGS_PATH}

# Path contains the svs slides
# This is just used for getting the height and width
# of the slides
SLIDES=${SVS_INPUT_PATH}
# If you want to test this script, use the following configuration:
#SLIDES=/data03/tcga_data/tumor/brca/

# delete all subfiles/subfolders in these below folders before saving new data
rm -r ./tumor_heatmaps/* ./tumor_image_to_extract/* ./tumor_ground_truth/*
rm -r ${TUMOR_HEATMAPS_PATH}/* ${TUMOR_IMAGES_TO_EXTRACT}/* ${TUMOR_GROUND_TRUTH}/*


for files in ${MARKING_FOLDER}/*_weight.txt; do
    # Get slide id
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'__x__' '{print $1}'`
    # Get user name
    USER=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'__x__' '{print $2}'`

    # Get corresponding marking files
    MARK=`echo ${files} | awk '{gsub("weight", "mark"); print $0;}'`

    SVS_FILE=`ls -1 ${SLIDES}/${SVS}*.svs | head -n 1`
    if [ ! -f ${SVS_FILE} ]; then
        echo ${SLIDES}/${SVS}.XXXX.svs does not exist.
        continue;
    fi
    WIDTH=` openslide-show-properties ${SVS_FILE} \
          | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SVS_FILE} \
          | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    matlab -nodisplay -singleCompThread -r \
    "get_tumor_pos_neg_map('${SVS}', '${USER}', ${WIDTH}, ${HEIGHT}, '${MARK}'); exit;" \
    </dev/null
done

cp ./tumor_heatmaps/* ${TUMOR_HEATMAPS_PATH}/
cp ./tumor_image_to_extract/* ${TUMOR_IMAGES_TO_EXTRACT}/
cp ./tumor_ground_truth/* ${TUMOR_GROUND_TRUTH}/

exit 0
