#!/bin/bash

source ../conf/variables.sh

FOLDER=../svs/
TYPE=${DEFAULT_TYPE}

sshpass -p "levu2016" ssh lehhou@osprey.bmi.stonybrook.edu "mkdir -p /home/data/tcga_data/${TYPE}/"
for files in ${FOLDER}/*; do
    sshpass -p "levu2016" scp ${files} lehhou@osprey.bmi.stonybrook.edu:/home/data/tcga_data/${TYPE}/
    sleep 2
done

echo ${0} DONE

exit 0
