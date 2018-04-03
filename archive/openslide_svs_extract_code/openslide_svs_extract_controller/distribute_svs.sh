#!/bin/bash

LINE_N=0
while read line; do
    LINE_N=$((LINE_N+1))
    if [ $((LINE_N % 9)) -eq 0 ]; then
        echo 001;
        scp ${line} lehhou@nfs001:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 1 ]; then
        echo 002;
        scp ${line} lehhou@nfs002:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 2 ]; then
        echo 003;
        scp ${line} lehhou@nfs003:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 3 ]; then
        echo 004;
        scp ${line} lehhou@nfs004:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 4 ]; then
        echo 005;
        scp ${line} lehhou@nfs005:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 5 ]; then
        echo 006;
        scp ${line} lehhou@nfs006:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 6 ]; then
        echo 007;
        scp ${line} lehhou@nfs007:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 7 ]; then
        echo 008;
        scp ${line} lehhou@nfs008:/data/shared/lehhou/openslide_svs_extract/svs/
    elif [ $((LINE_N % 9)) -eq 8 ]; then
        echo 010;
        scp ${line} lehhou@nfs010:/data/shared/lehhou/openslide_svs_extract/svs/
    fi
done < ${1}

exit 0
