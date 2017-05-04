#!/bin/bas

for files in /data07/tcga_data/tumor/coad/TCGA-??-????-???-??-DX*.svs; do
    svs=`echo ${files} | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}'`
    if [ ! -f ../rates-coad-all-auto/rate.${svs}.automatic.png ]; then
        echo ${svs}
    fi
done > missing_list.txt

exit 0
