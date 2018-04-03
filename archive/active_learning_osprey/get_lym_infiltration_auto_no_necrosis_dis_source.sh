#!/bin/bash

SLIDES=/data04/tcga_data/tumor/cesc/
PRED_F=patch-level-pred-4-25-blca_stad_cesc.txt

## useless conf
BOGUS_W=bogus/automatic_weight.txt
BOGUS_M=bogus/automatic_mark.txt
LIST_FILE=/tmp/svs_extract_patch_list.txt
ls -1d /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/TCGA-*/ > ${LIST_FILE}
## useless conf

for files in ${SLIDES}/TCGA-??-????-???-??-DX*.svs; do
    SVS=`echo ${files}      | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`
    SVS_FULL=`echo ${files} | awk -F'/' '{print $NF}'`

    WHITE=`grep ${SVS_FULL} ${LIST_FILE} | head -n 1`/patch-level-whiteness.txt
    if [ ! -f "${WHITE}" ]; then
        echo ${WHITE} not exist
        continue
    fi
    PRED=`grep ${SVS_FULL} ${LIST_FILE} | head -n 1`/${PRED_F}
    if [ ! -f "${PRED}" ]; then
        echo ${PRED} not exist
        continue
    fi

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    OUTF=rates/rate.${SVS}.automatic.png
    if [ -f "${OUTF}" ]; then
        echo ${OUTF} exists
        continue
    fi

    matlab -nodisplay -singleCompThread -r \
    "get_lym_infiltration_auto('${SVS}', ${WIDTH}, ${HEIGHT}, 'automatic', '${BOGUS_W}', '${BOGUS_M}', '${PRED}', '${WHITE}'); exit;" \
    </dev/null
done

#bash auto_thres.sh

exit 0
