#!/bin/bash

while read line; do
    for NFSN in 01 02 03 04 05 06 07 08 10; do
        cp /data${NFSN}/shared/lehhou/openslide_svs_extract/patches/${line}.*/* /data03/shared/lehhou/lym_project/svs_tiles/${line}.*/
    done
done < missing_list.txt

exit 0
