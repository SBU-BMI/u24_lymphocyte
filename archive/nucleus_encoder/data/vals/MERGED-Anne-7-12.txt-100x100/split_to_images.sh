#!/bin/bash

while read line; do
    PAT=`echo ${line} | awk '{print $1}'`
    LAB=`echo ${line} | awk '{print $2}'`
    IMG=`echo ${line} | awk '{print $3}'`
    FLD=`echo ${line} | awk '{print substr($3,9,12);}'`
    mkdir -p ${FLD}
    cp ${PAT} ${FLD}
    echo ${PAT} ${LAB} ${IMG} >> ${FLD}/label.txt
done < label.txt

exit 0
