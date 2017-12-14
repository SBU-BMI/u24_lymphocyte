#!/bin/bash

T=${1}

while read line; do
    sshpass -p "levu2016" scp ${line} lehhou@osprey.bmi.stonybrook.edu:/home/data/tcga_data/${T}/
    sleep 5
done < distribute_svs_${T}.txt
