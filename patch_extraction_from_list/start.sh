#!/bin/bash

source ../conf/variables.sh

INPUT_FILE=${1}
HEADER_LINE=${2}
SLIDE_ID=`echo ${INPUT_FILE} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`
SLIDE_USER=`echo ${INPUT_FILE} | awk -F'/' '{print $NF}'`

SLIDE_FOLDER=${SVS_INPUT_PATH}
#SLIDE_FOLDER=/data03/tcga_data/tumor/brca/
SLIDE_FILE=`ls -1 ${SLIDE_FOLDER}/${SLIDE_ID}*.svs | head -n 1`

if [ ! -f ${SLIDE_FILE} ]; then
    echo ${SLIDE_ID} does not exist under ${SLIDE_FOLDER}
    exit 1
fi

mkdir -p ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_USER}

awk -v header=${HEADER_LINE} 'NR>header' ${INPUT_FILE} | while read line; do
    ext_x0=`  echo ${line} | awk -F',' '{print int($1-2*($3-$1))}'`
    ext_y0=`  echo ${line} | awk -F',' '{print int($2-2*($4-$2))}'`
    ext_size=`echo ${line} | awk -F',' '{print int(5*($3-$1))}'`

    if [ ${ext_x0} -le 0 ]; then
        continue;
    fi
    if [ ${ext_y0} -le 0 ]; then
        continue;
    fi
    label=`echo ${line} | awk -F',' '{print $NF}'`
    openslide-write-png ${SLIDE_FILE} ${ext_x0} ${ext_y0} 0 ${ext_size} ${ext_size} \
        ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_USER}/${ext_x0}-${ext_y0}-${ext_size}-original_size.png
    convert ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_USER}/${ext_x0}-${ext_y0}-${ext_size}-original_size.png -resize 500x500 \
        ${PATCH_FROM_HEATMAP_PATH}/${SLIDE_USER}/${ext_x0}-${ext_y0}-${ext_size}-20X.png
done

exit 0
