#!/bin/bash

PARAL=$1
MAX_PARAL=$2

cd /data03/shared/lehhou/lym_project/

LINE_N=0
for files in /data/shared/lehhou/openslide_svs_extract/patches/TCGA-*; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    if [ -f ${files}/patch-level-whiteness.txt ]; then
        :
    else
        echo ${files}/patch-level-whiteness.txt generating whiteness
        python get_whiteness.py ${files}
    fi
done

exit 0
