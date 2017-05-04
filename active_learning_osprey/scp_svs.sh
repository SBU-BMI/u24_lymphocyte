#!/bin/bash

SLIDES=/data02/tcga_data/tumor/blca/
for files in ${SLIDES}/TCGA-??-????-???-??-DX*.svs; do
    sshpass -p scotthoule0312 scp ${files} lehou@129.49.249.175:/home/lehou/data/images/blca/
    sleep 2
done

SLIDES=/data02/tcga_data/tumor/stad/
for files in ${SLIDES}/TCGA-??-????-???-??-DX*.svs; do
    sshpass -p scotthoule0312 scp ${files} lehou@129.49.249.175:/home/lehou/data/images/stad/
    sleep 2
done

SLIDES=/data04/tcga_data/tumor/cesc/
for files in ${SLIDES}/TCGA-??-????-???-??-DX*.svs; do
    sshpass -p scotthoule0312 scp ${files} lehou@129.49.249.175:/home/lehou/data/images/cesc/
    sleep 2
done

echo scp_svs.sh DONE

exit 0
