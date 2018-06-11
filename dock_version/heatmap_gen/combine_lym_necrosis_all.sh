#!/bin/bash

source ../conf/variables.sh

LYM_FOLDER=./patch-level-lym/
NEC_FOLDER=./patch-level-nec/

for files in ${LYM_FOLDER}/*; do
    if [ ! -f ${files} ]; then continue; fi

    fn=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -f ${NEC_FOLDER}/${fn} ]; then
        bash combine_lym_necrosis.sh ${fn} &> ${LOG_OUTPUT_FOLDER}/log.combine_lym_necrosis.txt
    else
        bash combine_lym_no_necrosis.sh ${fn} &> ${LOG_OUTPUT_FOLDER}/log.combine_lym_no_necrosis.txt
    fi
done

python low_res_all.py

exit 0
