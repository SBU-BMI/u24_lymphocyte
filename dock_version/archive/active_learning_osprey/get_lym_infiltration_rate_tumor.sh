#!/bin/bash

SLIDES=/data06/tcga_data/tumor/ucec/
FOLDER=data-ucec-tumor-marker

LIST_FILE=/tmp/svs_extract_patch_list.txt
ls -1d /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/TCGA-*/ > ${LIST_FILE}

for files in ${FOLDER}/*_weight.txt; do
    SVS=`echo ${files} | awk -F'/|_' '{print $2}'`
    USER=`echo ${files} | awk -F'/|_' '{print $3}'`

    WEIGHT=${files}
    MARK=`echo ${files} | awk '{gsub("weight", "mark"); print $0;}'`
    PRED=`ls -1 ../heatmaps_v2/gened-v4/prediction-${SVS}*|grep -v low_res`
    if [ ! -f "${PRED}" ]; then
        echo PRED for ${SVS} not exist
        continue
    fi

    FULL_SVS=`echo ${PRED} | awk -F'/prediction-' '{print $2}'`
    WHITE_PATH=`grep ${FULL_SVS} ${LIST_FILE} | head -n 1`
    WHITE=${WHITE_PATH}/patch-level-whiteness.txt
    if [ ! -f "${WHITE}" ]; then
        echo ${WHITE} not exist
        continue
    fi

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    matlab -nodisplay -singleCompThread -r \
    "get_lym_infiltration_rate_tumor_pre_thres('${SVS}', ${WIDTH}, ${HEIGHT}, '${USER}', '${WEIGHT}', '${MARK}', '${PRED}', '${WHITE}'); exit;" \
    </dev/null
done

exit 0
