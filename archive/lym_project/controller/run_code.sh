#!/bin/bash

N=0
M=14
for NFSN in 002 003 004 005 006 007 008; do
    echo ${NFSN}

    ssh node${NFSN} "nohup bash /data03/shared/lehhou/lym_project/deep_conv_classification_alt48_adeno_t1_heatmap.sh ${N} ${M} &>> /data03/shared/lehhou/lym_project/log.adeno_t1_heatmap.txt &"
    N=$((N+1))
    sleep 1;
    ssh node${NFSN} "nohup bash /data03/shared/lehhou/lym_project/deep_conv_classification_alt48_adeno_t1_heatmap.sh ${N} ${M} &>> /data03/shared/lehhou/lym_project/log.adeno_t1_heatmap.txt &"
    N=$((N+1))
    sleep 1;
done

exit 0
