#!/bin/bash

while read line; do
    cp ${line}/label.txt ${line}/label.bak.6.txt
    NLINE=`cat ${line}/label.bak.6.txt | wc -l`
    SKIP=$((NLINE/500+1))
    echo $SKIP
    cat ${line}/label.bak.6.txt | awk -v skip=$SKIP 'NR%skip==0' > ${line}/label.txt
done < sample_label.txt

exit 0
