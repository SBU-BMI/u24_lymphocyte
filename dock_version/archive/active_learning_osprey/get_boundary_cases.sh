#!/bin/bash

SLIDES=/data06/tcga_data/tumor/ucec/
IMG_FOLDER=./rates/
SCP_F=boundary_cases_ucec

USE_PRED_F=0
PRED_FOLDER=../heatmaps_v2/gened/
PRED_F=patch-level-pred-3-21-paad.txt

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
    if [ ! -f ${IMG} ]; then
        echo ${IMG} not exist
        continue
    fi

    if [ ${USE_PRED_F} -eq 1 ]; then
        PRED=/data03/shared/lehhou/lym_project/svs_tiles/${SVS_FULL}/${PRED_F}
        if [ ! -f "${PRED}" ]; then
            echo ${PRED} not exist
            continue
        fi
    else
        if [ `ls -1 ${PRED_FOLDER}/prediction-${SVS}*|wc -l` -eq 0 ]; then
            echo ${SVS} not exist
            continue
        fi
        PRED=`ls -1 ${PRED_FOLDER}/prediction-${SVS}*|grep -v low_res`
    fi

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS_FULL} | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    if [ -f ${IMG} ]; then
        matlab -nodisplay -singleCompThread -r "get_boundary_cases('${SVS}', '${IMG}', '${PRED}', ${WIDTH}, ${HEIGHT}); exit;" </dev/null
    fi
done

sshpass -p scotthoule0312 scp -r boundary_cases lehou@129.49.249.175:/home/lehou/data/images/active_learning_osprey/extract_list/${SCP_F}

exit 0
