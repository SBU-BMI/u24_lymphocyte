#!/bin/bash

ls -l image/*.png | awk -F'/' '{print $NF}' > image.txt
ls -l mask/*.png | awk -F'/' '{print $NF}' > mask.txt

echo mask
awk 'NR==FNR{h[$1]=1;} NR!=FNR{if(!($1 in h)){print $1;}}' image.txt mask.txt > mask_del.txt
cat mask_del.txt
echo image
awk 'NR==FNR{h[$1]=1;} NR!=FNR{if(!($1 in h)){print $1;}}' mask.txt image.txt > image_del.txt
cat image_del.txt

while read line; do
    rm image/${line}
done < image_del.txt

while read line; do
    rm mask/${line}
done < mask_del.txt

ls -l image/*.png | awk '{print $NF}' > image.txt
awk 'NR==FNR{h[$1]=1;} NR!=FNR{if($1 in h){print $0;}}' image.txt image/label.txt > tttttmp
mv tttttmp image/label.txt
bash get_roundness.sh

rm image.txt mask.txt image_del.txt mask_del.txt

exit 0
