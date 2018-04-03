#!/bin/bash

CANCERTYPE=$1
LISTFILE=gen_humanmark_list.txt
LISTFILE_DONE=gen_humanmark_list_done.txt

awk 'NR==FNR{h[$0]=1} NR!=FNR{if(!($0 in h))print}' ${LISTFILE_DONE} ${LISTFILE} > /tmp/${LISTFILE}.tmp

python gen_humanmark_meta.py /tmp/${LISTFILE}.tmp

sshpass -p "levu2016" scp ./*.json lehhou@osprey.bmi.stonybrook.edu:/home/lehhou/humanmark_pipeline/
sshpass -p "levu2016" scp ./${LISTFILE} lehhou@osprey.bmi.stonybrook.edu:/home/lehhou/humanmark_pipeline/
sshpass -p "levu2016" ssh lehhou@osprey.bmi.stonybrook.edu "cd /home/lehhou/humanmark_pipeline/; bash put_humanmark.sh ${LISTFILE} u24_${CANCERTYPE}"

cat /tmp/${LISTFILE}.tmp >> ${LISTFILE_DONE}

rm *.json
