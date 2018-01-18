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

# Locations of unmodified heatmaps
# The filenames of the unmodifed heatmaps should be:
#   prediction-${slide_id}
# For example:
#   prediction-TCGA-NJ-A55O-01Z-00-DX1
HEAT_LOC=${HEATMAP_TXT_OUTPUT_FOLDER}
# If you want to test this script, use the following configuration:
#HEAT_LOC=/data08/shared/lehhou/heatmaps_v2/gened-all-luad-brca/

for files in ${MARKING_FOLDER}/*_weight.txt; do
    # Get slide id
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}'`
    # Get user name
    USER=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'_' '{print $2}'`

    # Get corresponding weight and marking files
    WEIGHT=${files}
    MARK=`echo ${files} | awk '{gsub("weight", "mark"); print $0;}'`

    # Find the unmodified heatmap
    PRED=`ls -1 ${HEAT_LOC}/prediction-${SVS}*|grep -v low_res`

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
    "get_modified_heatmap('${SVS}', ${WIDTH}, ${HEIGHT}, '${USER}', '${WEIGHT}', '${MARK}', '${PRED}'); exit;" \
    </dev/null
done

cp ./modified_heatmaps/* ${MODIFIED_HEATMAPS_PATH}/

exit 0
