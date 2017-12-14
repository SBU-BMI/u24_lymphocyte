#!/bin/bash

source ../conf/variables.sh

CANCERTYPE=${CANCER_TYPE}
LISTFILE=gen_humanmark_list.txt
LISTFILE_DONE=gen_humanmark_list_done.txt
SLIDES=../svs/

ls -1 ../svs/* | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}' | sort -u > ${LISTFILE}
awk 'FILENAME=="'${LISTFILE_DONE}'"{h[$0]=1} FILENAME=="'${LISTFILE}'"{if(!($0 in h))print}' ${LISTFILE_DONE} ${LISTFILE} > /tmp/${LISTFILE}.tmp

python gen_humanmark_meta.py /tmp/${LISTFILE}.tmp

for files in ./*.json; do
    if [ -f ${files} ]; then continue; fi
    bash ../util/scp_to_remote_camic.sh ${files} /home/lehhou/humanmark_pipeline/
    sleep 2
done
bash ../util/scp_to_remote_camic.sh ${LISTFILE} /home/lehhou/humanmark_pipeline/
bash ../util/operation_on_remote_camic.sh "cd /home/lehhou/humanmark_pipeline/; bash put_humanmark.sh ${LISTFILE} u24_${CANCERTYPE}"

cat /tmp/${LISTFILE}.tmp >> ${LISTFILE_DONE}

rm -rf *.json
