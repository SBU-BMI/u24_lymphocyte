#!/bin/bash

SLIDES=/data08/shared/lehhou/tcga/tumor/read/
PRED_F=patch-level-pred-3-21-paad.txt
FOLDER=data-read-15

for files in ${FOLDER}/*_weight.txt; do
    SVS=`echo ${files} | awk -F'/|_' '{print $2}'`
    USER=`echo ${files} | awk -F'/|_' '{print $3}'`
    SVS_FULL=`ls -l ${SLIDES}/TCGA-??-????-???-??-DX*.svs | grep ${SVS} | awk -F'/' '{print $NF}' | head -n 1`

    WHITE=/data03/shared/lehhou/lym_project/svs_tiles/${SVS_FULL}/patch-level-whiteness.txt
    if [ ! -f "${WHITE}" ]; then
        echo ${WHITE} not exist
        continue
    fi
    PRED=/data03/shared/lehhou/lym_project/svs_tiles/${SVS_FULL}/${PRED_F}
    if [ ! -f "${PRED}" ]; then
        echo ${PRED} not exist
        continue
    fi

    WEIGHT=${files}
    MARK=`echo ${files} | awk '{gsub("weight", "mark"); print $0;}'`

    if [ -f tumor_nontumor_maps/map.${SVS}.${USER}.png ]; then
        continue;
    fi

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    matlab -nodisplay -singleCompThread -r \
    "get_tumor_pos_neg_map('${SVS}', ${WIDTH}, ${HEIGHT}, '${USER}', '${WEIGHT}', '${MARK}', '${PRED}', '${WHITE}'); exit;" \
    </dev/null
done

exit 0
