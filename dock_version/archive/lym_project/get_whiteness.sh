#!/bin/bash

PARAL=$1
MAX_PARAL=$2

LINE_N=0
for files in svs_tiles/TCGA-*; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    if [ -f ${files}/patch-level-whiteness.txt ]; then
        :
    else
        echo ${files}/patch-level-whiteness.txt generating whiteness
        python get_whiteness.py ${files}
    fi

    #echo ${files}/patch-level-whiteness.txt generating whiteness
    #python get_whiteness.py ${files}
done

exit 0
