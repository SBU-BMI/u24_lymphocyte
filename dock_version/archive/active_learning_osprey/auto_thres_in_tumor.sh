#!/bin/bash

FOLDER=rates-ucec-all-auto
OUT_FOLD=rates-ucec-all-final
TUMOR_IM=rates-ucec-tumor-marker-to-use
FINAL_OUT_FOLD=rates-ucec-all-final-intumor

matlab -nodisplay -singleCompThread -r "auto_thres('${FOLDER}'); exit;" </dev/null

mkdir -p ${OUT_FOLD}
for files in ${FOLDER}/*automatic_thres.png; do
    fn=`echo ${files} | awk -F'.' '{print $2}'`
    cp ${files} ${OUT_FOLD}/${fn}.png
done

echo "Slides,TIL" > ${OUT_FOLD}/infiltration_rate.txt
cat ${FOLDER}/infiltration_rate.txt | awk '{print $1","$3}' >> ${OUT_FOLD}/infiltration_rate.txt

matlab -nodisplay -singleCompThread -r "remove_necrosis('${OUT_FOLD}'); exit;" </dev/null

mkdir ${FINAL_OUT_FOLD}
matlab -nodisplay -singleCompThread -r "rate_in_tumor('${OUT_FOLD}', '${TUMOR_IM}', '${FINAL_OUT_FOLD}'); exit;" </dev/null

exit 0
