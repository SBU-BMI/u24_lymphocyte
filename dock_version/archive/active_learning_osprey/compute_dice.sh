#!/bin/bash

while read slide; do
    while read user1; do
        while read user2; do
            if [ $user1 != $user2 ]; then
                svs=`echo ${slide} | awk -F'TCGA' '{print "TCGA"$2;}'`
                F1=${slide}.${user1}.png
                F2=${slide}.${user2}.png
                matlab -nodisplay -singleCompThread -r "compute_dice('${svs}', '${user1}', '${user2}', '${F1}', '${F2}'); exit;" </dev/null
            fi
        done < dice_user_list.txt
    done < dice_user_list.txt
done < dice_slides_list.txt

