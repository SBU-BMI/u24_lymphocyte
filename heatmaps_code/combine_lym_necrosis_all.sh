#!/bin/bash

for files in patch-level/*; do
    fn=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -f necrosis/${fn} ]; then
        if [ -f gened/${fn} ]; then
            :
        else
            echo ${fn}
            bash combine_lym_necrosis_use_this.sh ${fn}
        fi
    else
        bash combine_lym_no_necrosis.sh ${fn}
    fi
done

matlab -nodisplay -singleCompThread -r "low_res_all; exit;" </dev/null

exit 0
