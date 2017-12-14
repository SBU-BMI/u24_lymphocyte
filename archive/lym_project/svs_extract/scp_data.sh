#!/bin/bash

for files in /data03/tcga_data/tumor/brca/TCGA-???????????????D*.svs; do
    sshpass -p scotthoule0312 scp ${files} lehou@129.49.249.175:/home/lehou/data/images/brca/
    if [ $? -eq 0 ]; then
        echo ${files} done
    else
        echo ${files} error
    fi
    sleep 3
done

exit 0
