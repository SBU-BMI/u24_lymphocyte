#!/bin/bash

echo "INPUT:"
ls -1 /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/svs/ | grep TCGA | wc -l

echo "OUTPUT:"
ls -1 /data{01,02,03,04,05,06,07,08,10}/shared/lehhou/openslide_svs_extract/patches/ | grep TCGA | wc -l

echo
for NFSN in 01 02 03 04 05 06 07 08 10; do
    NOUT=`ls -1 /data${NFSN}/shared/lehhou/openslide_svs_extract/patches/ | grep TCGA | wc -l`
    TIMES=`ls -lt /data01/shared/lehhou/openslide_svs_extract/patches/ | grep TCGA | head -n 1 | awk '{print $6,$7,$8}'`
    echo data${NFSN}, ${NOUT}, ${TIMES}
done

exit 0
