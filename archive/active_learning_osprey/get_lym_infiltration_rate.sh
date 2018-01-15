#!/bin/bash

SLIDES=/data01/tcga_data/tumor/skcm/
HEAT_LOC=../heatmaps_v2/gened
FOLDER=data

for files in ${FOLDER}/*_weight.txt; do
    SVS=`echo ${files} | awk -F'/|_' '{print $2}'`
    USER=`echo ${files} | awk -F'/|_' '{print $3}'`

    WEIGHT=${files}
    MARK=`echo ${files} | awk '{gsub("weight", "mark"); print $0;}'`
    PRED=`ls -1 ${HEAT_LOC}/prediction-${SVS}*|grep -v low_res`
    WHITE=`echo ${PRED} | awk -F'/prediction-' '{print "/data03/shared/lehhou/lym_project/svs_tiles/"$2".svs/patch-level-whiteness.txt"}'`

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    if [ ! -f ${WHITE} ]; then
        echo ${WHITE} not exist
        continue
    fi

    echo \
    "get_lym_infiltration_rate('${SVS}', ${WIDTH}, ${HEIGHT}, '${USER}', '${WEIGHT}', '${MARK}', '${PRED}', '${WHITE}'); exit;"
    matlab -nodisplay -singleCompThread -r \
    "get_lym_infiltration_rate('${SVS}', ${WIDTH}, ${HEIGHT}, '${USER}', '${WEIGHT}', '${MARK}', '${PRED}', '${WHITE}'); exit;" \
    </dev/null
done

exit 0
