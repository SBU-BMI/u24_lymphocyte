#!/bin/bash

for files in NucleiInRegionMask/*.png; do
    p=`echo ${files} | awk -F'/|.png' '{print $2}'`
    mv NucleiInRegionMask/${p}.png tt/${p}_3.png
done

exit
