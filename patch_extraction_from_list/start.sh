#!/bin/bash

source ../conf/variables.sh

INPUT_FILE=${1}
HEADER_LINE=${2}
SLIDE_ID=`echo ${INPUT_FILE} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`

SLIDE_FOLDER=${SVS_INPUT_PATH}
#SLIDE_FOLDER=/data03/tcga_data/tumor/brca/

if [ ! `ls -1 ${SLIDE_FOLDER}/${SLIDE_ID}*.svs` ]; then
    echo "${SLIDE_ID}*.svs does not exist under ${SLIDE_FOLDER}. Trying tif..."
    SLIDE_FILE=`ls -1 ${SLIDE_FOLDER}/${SLIDE_ID}*.tif | head -n 1`
else
    SLIDE_FILE=`ls -1 ${SLIDE_FOLDER}/${SLIDE_ID}*.svs | head -n 1`
fi

if [ -z "$SLIDE_FILE" ]; then
    echo "Could not find slide."
    exit 1
fi

#rm -f ${PATCH_FROM_HEATMAP_PATH}/label.txt
awk -v header=${HEADER_LINE} 'NR>header' ${INPUT_FILE} | while read line; do
    ext_x0=`  echo ${line} | awk -F',' '{print int($1-2*($3-$1))}'`
    ext_y0=`  echo ${line} | awk -F',' '{print int($2-2*($4-$2))}'`
    ext_size=`echo ${line} | awk -F',' '{print int(5*($3-$1))}'`
    #ext_x0=`  echo ${line} | awk -F',' '{print int($1)}'`
    #ext_y0=`  echo ${line} | awk -F',' '{print int($2)}'`
    #ext_size=`echo ${line} | awk -F',' '{print int($3-$1)}'`
    label=`echo ${line} | awk -F',' '{print $NF}'`

    if [ ${ext_x0} -le 0 ]; then
        continue;
    fi
    if [ ${ext_y0} -le 0 ]; then
        continue;
    fi
    if [ ${ext_size} -le 0 ]; then
        continue;
    fi

    RESIZE=`bash get_mpp_w_h.sh ${SLIDE_FILE} | awk -v s=${ext_size} '{print int(s*$1/0.50+0.5)}'`
    #RESIZE=500
    openslide-write-png \
        ${SLIDE_FILE} ${ext_x0} ${ext_y0} 0 ${ext_size} ${ext_size} \
        ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_ID}-${ext_x0}-${ext_y0}-${ext_size}-original_size.png
    if [ $? -eq 0 ]; then
        convert \
            ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_ID}-${ext_x0}-${ext_y0}-${ext_size}-original_size.png \
            -resize ${RESIZE}x${RESIZE} \
            ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_ID}-${ext_x0}-${ext_y0}-${ext_size}-20X.png
        echo ${SLIDE_ID}-${ext_x0}-${ext_y0}-${ext_size}-20X.png ${label} >> ${PATCH_FROM_HEATMAP_PATH}/label.txt
    fi
done

exit 0
