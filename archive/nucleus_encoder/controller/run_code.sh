#!/bin/bash

N=0
for NFSN in 002 003 004 005 006 007 008 009 010; do
    echo ${NFSN}

    ssh node${NFSN} "nohup bash /data08/shared/lehhou/nucleus_encoder/run_nec_le_hou.sh ${N} 18 &>> /data08/shared/lehhou/nucleus_encoder/log.run_nec_le_hou.txt &"
    N=$((N+1))
    sleep 1;
    ssh node${NFSN} "nohup bash /data08/shared/lehhou/nucleus_encoder/run_nec_le_hou.sh ${N} 18 &>> /data08/shared/lehhou/nucleus_encoder/log.run_nec_le_hou.txt &"
    N=$((N+1))
    sleep 1;
done

exit 0
