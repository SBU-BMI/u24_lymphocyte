#!/bin/bash

for files in *.azhao83.txt; do
    SVS=`echo ${files} | awk -F'.' '{print $2}'`
    awk -f merge_agree.awk log.${SVS}.azhao83.txt log.${SVS}.john.vanarnam.txt > log.${SVS}.agree.txt
done

exit 0
