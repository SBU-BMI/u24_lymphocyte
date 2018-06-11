#!/bin/bash

for NFSN in 001 002 003 004 005 006 007 008 010; do
    echo ${NFSN}

    ssh nfs${NFSN} "nohup bash /data03/shared/lehhou/lym_project/get_whiteness_on_nfs.sh 0 2 &>> /data03/shared/lehhou/lym_project/log.whiteness.txt &"
    sleep 1;
    ssh nfs${NFSN} "nohup bash /data03/shared/lehhou/lym_project/get_whiteness_on_nfs.sh 1 2 &>> /data03/shared/lehhou/lym_project/log.whiteness.txt &"
    sleep 1;
done

exit 0
