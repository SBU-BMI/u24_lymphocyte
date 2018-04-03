#!/bin/bash

PARAL=$1
MAX_PARAL=$2

LINE_N=0
while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi
    files=`ls -1d svs_tiles/${line}*`

    echo ${files}/patch-level-whiteness.txt generating whiteness_from_list
    python get_whiteness.py ${files}
done < coad_missing_list.txt

exit 0
