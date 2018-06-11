#!/bin/bash

source ~/.bashrc_theano

# PARAL = [0, MAX_PARAL-1]
PARAL=$1
MAX_PARAL=$2
GPU=$((PARAL % 2))

echo staring ${PARAL} ${MAX_PARAL} ${GPU}
cd /data03/shared/lehhou/lym_project/
LINE_N=0

LIST_FILE=/tmp/openslide_svs_extract_patches_list_${PARAL}.txt
ls -1d /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/*/ > ${LIST_FILE}

while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % MAX_PARAL)) -ne ${PARAL} ]; then continue; fi

    EXIST=`grep "/${line}/" ${LIST_FILE} | wc -l`
    if [ ${EXIST} -eq 0 ]; then continue; fi

    FULL_PATH=`grep "/${line}/" ${LIST_FILE}`
    echo doing ${PARAL} ${MAX_PARAL} ${GPU} ${FULL_PATH}
    THEANO_FLAGS="device=gpu${GPU}" python -u deep_conv_classification_alt48_adeno_t1_heatmap.py ${FULL_PATH}
done < read.txt

exit 0
