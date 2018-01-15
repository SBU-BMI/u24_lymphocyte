#!/bin/bash

LYM_FOLDER=./patch-level-lym/
NEC_FOLDER=./patch-level-nec/

for files in ${LYM_FOLDER}/*; do
    if [ ! -f ${files} ]; then continue; fi

    fn=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -f ${NEC_FOLDER}/${fn} ]; then
        bash combine_lym_necrosis.sh ${fn}
    else
        bash combine_lym_no_necrosis.sh ${fn}
    fi
done

matlab -nodisplay -singleCompThread -r "low_res_all; exit;" </dev/null

exit 0
