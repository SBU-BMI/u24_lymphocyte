#!/bin/bash

cat list.txt | while read svs; do
    ID=`echo ${svs}|awk -F'/' '{print substr($NF,1,length($NF)-4)}'`
    N_LVL=`openslide-show-properties ${svs}|grep openslide.level-count|awk '{print substr($NF,2,length($NF)-2)}'`
    WIDTH=`openslide-show-properties ${svs}|grep "openslide.level\[$((N_LVL-1))\].width"|awk '{print substr($NF,2,length($NF)-2)}'`
    HEIGHT=`openslide-show-properties ${svs}|grep "openslide.level\[$((N_LVL-1))\].height"|awk '{print substr($NF,2,length($NF)-2)}'`
    openslide-write-png ${svs} 1 1 $((N_LVL-1)) $WIDTH $HEIGHT ${ID}.png
    convert ${ID}.png -resize $((WIDTH/6))x$((HEIGHT/6)) tmp.png
    mv tmp.png ${ID}.png
done

exit 0
