#!/bin/bash

SLIDES=/data06/tcga_data/tumor/ucec/
IMG_FOLDER=./rates-ucec-tumor-marker-dis-0.015
SCP_F=boundary_cases_ucec

PRED_F=patch-level-pred-4-14-ucec.txt
LIST_FILE=/tmp/svs_extract_patch_list.txt
ls -1d /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/TCGA-*/ > ${LIST_FILE}

rm -rf boundary_cases
mkdir boundary_cases
for files in ${SLIDES}/TCGA-??-????-???-??-DX*.svs; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`
    SVS_FULL=`echo ${files} | awk -F'/' '{print $NF}'`

    if [ -f boundary_cases/log.${SVS}.boundary_cases.txt ]; then
        echo ${SVS} already done
        continue;
    fi

    IMG=`ls -1 ${IMG_FOLDER}/rate.${SVS}.*.png | head -n 1`
    if [ ! -f "${IMG}" ]; then
        echo ${IMG} not exist
        continue
    fi

    PRED=`grep ${SVS_FULL} ${LIST_FILE} | head -n 1`/${PRED_F}
    if [ ! -f "${PRED}" ]; then
        echo ${PRED} not exist
        continue
    fi

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    if [ -f ${IMG} ]; then
        matlab -nodisplay -singleCompThread -r "get_boundary_cases_in_tumor('${SVS}', '${IMG}', '${PRED}', ${WIDTH}, ${HEIGHT}); exit;" </dev/null
    fi
done

sshpass -p scotthoule0312 scp -r boundary_cases lehou@129.49.249.175:/home/lehou/data/images/active_learning_osprey/extract_list/${SCP_F}

exit 0
