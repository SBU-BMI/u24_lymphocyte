#!/bin/bash

#for files in rates-luad-all-auto/*automatic_thres.png; do
#    fn=`echo ${files} | awk -F'.' '{print $2}'`
#    cp ${files} rates-luad-final/${fn}.png
#done

#for files in rates-luad-184/*.png; do
#    fn=`echo ${files} | awk -F'.' '{print $2}'`
#    cp ${files} rates-luad-final/${fn}.png
#done

#for files in rates-luad-15-questionmarks/*.png; do
#    fn=`echo ${files} | awk -F'.' '{print $2}'`
#    cp ${files} rates-luad-final/${fn}.png
#done

for files in rates-brca-all-auto/*automatic_thres.png; do
    fn=`echo ${files} | awk -F'.' '{print $2}'`
    cp ${files} rates-brca-all-final/${fn}.png
done

exit 0
