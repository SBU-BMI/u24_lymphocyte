#!/bin/bash

source ../conf/variables.sh

SLIDES=${SVS_INPUT_PATH}
HEAT_PNG_LOC=${GRAYSCALE_HEATMAPS_PATH}
HEAT_TXT_LOC=${HEATMAP_TXT_OUTPUT_FOLDER}

for file in ${HEAT_PNG_LOC}/*.png; do
    SVS=`echo ${file} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`
    SVS_FILE=`ls -1 ${SLIDES}/${SVS}*.svs | head -n 1`
    if [ ! -f ${SVS_FILE} ]; then
        echo ${SLIDES}/${SVS}.XXXX.svs does not exist.
        continue;
    fi

    HEAT_PNG=${file}
    if [ ! -f ${HEAT_PNG} ]; then
        echo ${HEAT_PNG} not exist
        continue
    fi

    PRED=`ls -1 ${HEAT_TXT_LOC}/prediction-${SVS}* | grep -v low_res | head -n 1`
    if [ ! -f "${PRED}" ]; then
        echo ${PRED} not exist
        continue
    fi

    WIDTH=` openslide-show-properties ${SVS_FILE} \
          | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SVS_FILE} \
          | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    matlab -nodisplay -singleCompThread -r \
        "get_sample_list('${SVS}', '${HEAT_PNG}', '${PRED}', ${WIDTH}, ${HEIGHT}); exit;" </dev/null
    
    #python -u get_sample_list.py ${SVS} ${HEAT_PNG} ${PRED} ${WIDTH} ${HEIGHT}
done

cp sample_list/* ${PATCH_SAMPLING_LIST_PATH}/

exit 0
