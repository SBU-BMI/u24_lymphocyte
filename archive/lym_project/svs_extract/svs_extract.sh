#!/bin/bash

for files in /data07/tcga_data/tumor/coad/TCGA-???????????????D*.svs; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -d ../svs_tiles/${SVS} ]; then
        echo ${SVS} existed
    else
        echo ${SVS} extract
        sshpass -p scotthoule0312 scp ${files} lehou@129.49.249.175:/home/lehou/data/images/coad/
        sleep 2
    fi
done

#sshpass -p scotthoule0312 ssh lehou@129.49.249.175 touch /home/lehou/data/images/svs_extract/sig_file

exit 0
