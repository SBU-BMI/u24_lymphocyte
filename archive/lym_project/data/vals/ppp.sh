#!/bin/bash

while read line; do
    SVS=`echo ${line} | awk -F'.' '{print $2}'`
    cp -r ${line} log.${SVS}.agree.txt
    awk '
        NR==FNR{
        k = $2"\t"$3"\t"$4"\t"$5"\t"$6;
        h[k] = 1;
    }

    NR!=FNR{
        k = $2"\t"$3"\t"$4"\t"$5"\t"$6;
        if (k in h) {
            print
        }
    }' log.${SVS}.john.vanarnam.txt/label.txt log.${SVS}.agree.txt/label.txt > log.${SVS}.agree.txt/tmp.txt
    mv log.${SVS}.agree.txt/tmp.txt log.${SVS}.agree.txt/label.txt
done < ppp.txt

exit 0
