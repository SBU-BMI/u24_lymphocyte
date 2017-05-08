#!/bin/bash

SLIDES=/data06/tcga_data/tumor/ucec/
#SLIDES=/data02/tcga_data/tumor/lusc/
#SLIDES=/data08/tcga_data/tumor/prad/
HEAT_LOC=../heatmaps_v2/gened-v4/
PRED2_F=patch-level-pred-4-14-ucec.txt
PRED2_AVG=0

## useless conf
BOGUS_W=bogus/automatic_weight.txt
BOGUS_M=bogus/automatic_mark.txt
LIST_FILE=/tmp/svs_extract_patch_list.txt
ls -1d /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/TCGA-*/ > ${LIST_FILE}
## useless conf

for files in ${SLIDES}/TCGA-??-????-???-??-DX*.svs; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`

#ls -1 data-brca-12/*_weight.txt | awk -F'/|_' '{print $2}' | sort -u | while read files; do
#    SVS=${files}

    if [ -f rates/rate.${SVS}.automatic.png ]; then
        echo ${SVS} already done
        continue;
    fi

    if [ `ls -1 ${HEAT_LOC}/prediction-${SVS}*|wc -l` -eq 0 ]; then
        echo ${SVS} not exist
        continue
    fi
    PRED1=`ls -1 ${HEAT_LOC}/prediction-${SVS}*|grep -v low_res`

    FULL_SVS=`echo ${PRED1} | awk -F'/prediction-' '{print $2}'`
    WHITE_PATH=`grep ${FULL_SVS} ${LIST_FILE} | head -n 1`
    WHITE=${WHITE_PATH}/patch-level-whiteness.txt
    if [ ! -f "${WHITE}" ]; then
        echo ${WHITE} not exist
        continue
    else
        echo ${WHITE} exist
    fi

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    if [ ${PRED2_AVG} -eq 1 ]; then
        PRED2=`ls -1 /data03/shared/lehhou/lym_project/svs_tiles/${SVS}*/${PRED2_F}`
        awk -f combine_pred1_pred2.awk ${PRED1} ${PRED2} > /tmp/combined_heatmap_${SVS}
        PRED=/tmp/combined_heatmap_${SVS}
    else
        PRED=${PRED1}
    fi

    matlab -nodisplay -singleCompThread -r \
    "get_lym_infiltration_auto('${SVS}', ${WIDTH}, ${HEIGHT}, 'automatic', '${BOGUS_W}', '${BOGUS_M}', '${PRED}', '${WHITE}'); exit;" \
    </dev/null
done

#bash auto_thres.sh

exit 0
