#!/bin/bash

while read line; do
    iid=`echo ${line} | awk '{print $1}'`
    cid=`echo ${line} | awk '{print substr($2,1,length($2)-4)}'`
    lab=`cat label_files/im_id_${iid}.txt | awk '
BEGIN{
h["c1"]="None";
h["c2"]="Non-Brisk Focal";
h["c3"]="Non-Brisk Multifocal";
h["c4"]="Brisk Band-like";
h["c5"]="Brisk Diffuse";
h["c6"]="Borderline";
h["c7"]="Indeterminate";
}
{
print h[$1];
}'`
    echo ${cid}","${lab}
done < images/info.txt > compile_label.csv

awk 'NR==FNR{h[$1]=substr($2,1,length($2)-4);} NR!=FNR{print h[$1]}' images/info.txt unreliable_list_iid.txt > unreliable_list_cid.txt
awk -F',' 'BEGIN{print "Slides,PatternLabels,TIL-On-Ink-Artifacts"} NR==FNR{h[$1]=1} NR!=FNR{if($1 in h){print $1","$2",Y"}else{print $1","$2",-"}}' unreliable_list_cid.txt compile_label.csv > asdf
mv asdf compile_label.csv

exit 0
