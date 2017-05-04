#!/bin/bash

for files in /data08/tcga_data/tumor/uvm/TCGA-???????????????D*.svs; do
    SVS=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -d ../svs_tiles/${SVS} ]; then
        :
    else
        EXT=`awk -v svs=${SVS} 'BEGIN{ext=0} {if(match(svs, $1)){ext=1}} END{print ext;}' svs_extract_list.txt`
        if [ $EXT -eq 1 ]; then
            echo ${SVS} extract
            sshpass -p scotthoule0312 scp ${files} lehou@129.49.249.175:/home/lehou/data/images/uvm/
            sleep 2
        fi
    fi
done

sshpass -p scotthoule0312 ssh lehou@129.49.249.175 touch /home/lehou/data/images/svs_extract/sig_file

exit 0
