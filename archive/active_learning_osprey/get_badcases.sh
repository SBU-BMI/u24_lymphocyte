#!/bin/bash

SLIDES=/data08/tcga_data/tumor/prad/
FOLDER=data-prad-10
PRED_FOLDER=../heatmaps_v2/gened/
OUT_FOLDER=prad-10

for files in ${FOLDER}/*_weight.txt; do
    SVS=`echo ${files} | awk -F'/|_' '{print $2}'`
    USER=`echo ${files} | awk -F'/|_' '{print $3}'`

    WEIGHT=${files}
    MARK=`echo ${files} | awk '{gsub("weight", "mark"); print $0;}'`
    PRED=`ls -1 ${PRED_FOLDER}/prediction-${SVS}*|grep -v low_res`

    WIDTH=` openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].width"  | awk '{print substr($2,2,length($2)-2);}'`
    HEIGHT=`openslide-show-properties ${SLIDES}/${SVS}*.svs | grep "openslide.level\[0\].height" | awk '{print substr($2,2,length($2)-2);}'`

    matlab -nodisplay -singleCompThread -r \
        "get_badcases('${SVS}', ${WIDTH}, ${HEIGHT}, '${USER}', '${WEIGHT}', '${MARK}', '${PRED}'); exit;" </dev/null
done

sshpass -p scotthoule0312 scp -r logs lehou@129.49.249.175:/home/lehou/data/images/active_learning_osprey/extract_list/${OUT_FOLDER}

exit 0
