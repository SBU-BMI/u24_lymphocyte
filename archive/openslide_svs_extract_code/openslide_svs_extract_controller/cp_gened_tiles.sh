#!/bin/bash

sleep 36000

for NFSN in 01 02 03 04 05 06 07 08 10; do
    echo "cp_gened_tiles ${NFSN}"
    date
    cd /data${NFSN}/shared/lehhou/openslide_svs_extract/patches/
    ls -ltd TCGA* | head -n 100 | awk '{print $NF}' > /tmp/svs_list.txt

    while read line; do
        cp -r ${line} /data03/shared/lehhou/lym_project/svs_tiles/
    done < /tmp/svs_list.txt
done

exit 0
