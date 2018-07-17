#!/bin/bash

source ../conf/variables.sh

INPUT_FILE=${1}
HEADER_LINE=${2}
SLIDE_ID=`echo ${INPUT_FILE} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`

SLIDE_FOLDER=${SVS_INPUT_PATH}
#SLIDE_FOLDER=/data03/tcga_data/tumor/brca/
SLIDE_FILE=`ls -1 ${SLIDE_FOLDER}/${SLIDE_ID}*.svs | head -n 1`

if [ ! -f ${SLIDE_FILE} ]; then
    echo ${SLIDE_ID} does not exist under ${SLIDE_FOLDER}
    exit 1
fi

#rm -f ${PATCH_FROM_HEATMAP_PATH}/label.txt
awk -v header=${HEADER_LINE} 'NR>header' ${INPUT_FILE} | while read line; do
    ext_x0=`  echo ${line} | awk -F',' '{print int($1-2*($3-$1))}'`
    ext_y0=`  echo ${line} | awk -F',' '{print int($2-2*($4-$2))}'`
    ext_size=`echo ${line} | awk -F',' '{print int(5*($3-$1))}'`  #original 5*int() to generate larger tiles
    #ext_x0=`  echo ${line} | awk -F',' '{print int($1)}'`
    #ext_y0=`  echo ${line} | awk -F',' '{print int($2)}'`
    #ext_size=`echo ${line} | awk -F',' '{print int($3-$1)}'`
    label=`echo ${line} | awk -F',' '{print int($NF)}'`

    if [ ${ext_x0} -le 0 ]; then
        continue;
    fi
    if [ ${ext_y0} -le 0 ]; then
        continue;   
    fi
    if [ ${ext_size} -le 0 ]; then
        continue;
    fi

    #RESIZE=`bash get_mpp_w_h.sh ${SLIDE_FILE} | awk -v s=${ext_size} '{print int(s*$1/0.50+0.5)}'`
    RESIZE=300
    if [ ${ext_size} -lt $RESIZE ]; then
        continue;
    fi
    
    echo "after ground check"
    img_file=${SLIDE_ID}-${ext_x0}-${ext_y0}-${ext_size}-20X-${label}.png
    img_file_12X=${SLIDE_ID}-${ext_x0}-${ext_y0}-${ext_size}-12X-${label}.png
    openslide-write-png \
        ${SLIDE_FILE} ${ext_x0} ${ext_y0} 0 ${ext_size} ${ext_size} \
        ${PATCH_FROM_HEATMAP_PATH}/${img_file}
    if [ $? -eq 0 ]; then
        echo "start procedure"
        # do something to delete white patches
        # check if it is white patch only for label -1 (background)
        if [ ${label} -lt 0 ]; then     # background patches
            python -u img_stats.py ${PATCH_FROM_HEATMAP_PATH}/${img_file} 230 5
            if [ -f 'isWhitePatch.txt' ]; then
                rm -rf ${PATCH_FROM_HEATMAP_PATH}/${img_file}
            fi
        else    # if label 0/1, meaningful patches
            python -u img_stats.py ${PATCH_FROM_HEATMAP_PATH}/${img_file} 230 20
            if [ -f 'isWhitePatch.txt' ]; then
                rm -rf ${PATCH_FROM_HEATMAP_PATH}/${img_file}
            fi
        fi
        if [ -f ${PATCH_FROM_HEATMAP_PATH}/${img_file} ]; then
            echo "start convert"
            convert \
                ${PATCH_FROM_HEATMAP_PATH}/${img_file} \
                -resize ${RESIZE}x${RESIZE} \
                ${PATCH_FROM_HEATMAP_PATH}/${img_file_12X}
            echo ${img_file} ${label} ${SLIDE_ID} >> ${PATCH_FROM_HEATMAP_PATH}/label.txt
            echo ${img_file_12X} ${label} ${SLIDE_ID} >> ${PATCH_FROM_HEATMAP_PATH}/label.txt
        fi

    fi

done

exit 0
