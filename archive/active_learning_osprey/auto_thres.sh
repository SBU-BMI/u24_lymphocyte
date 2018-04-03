#!/bin/bash

FOLDER=rates-prad-all-auto
OUT_FOLD=rates-prad-all-final

matlab -nodisplay -singleCompThread -r "auto_thres('${FOLDER}'); exit;" </dev/null

mkdir -p ${OUT_FOLD}
for files in ${FOLDER}/*automatic_thres.png; do
    fn=`echo ${files} | awk -F'.' '{print $2}'`
    cp ${files} ${OUT_FOLD}/${fn}.png
done

echo "Slides,TIL" > ${OUT_FOLD}/infiltration_rate.txt
cat ${FOLDER}/infiltration_rate.txt | awk '{print $1","$3}' >> ${OUT_FOLD}/infiltration_rate.txt

matlab -nodisplay -singleCompThread -r "remove_necrosis('${OUT_FOLD}'); exit;" </dev/null

exit 0
