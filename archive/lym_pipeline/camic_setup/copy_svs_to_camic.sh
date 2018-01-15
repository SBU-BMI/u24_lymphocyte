#!/bin/bash

source ../conf/variables.sh

FOLDER=../svs/
TYPE=${DEFAULT_TYPE}

bash ../util/operation_on_remote_camic.sh "mkdir -p /home/data/tcga_data/${TYPE}/"
for files in ${FOLDER}/*; do
    bash ../util/scp_to_remote_camic.sh ${files} /home/data/tcga_data/${TYPE}/
    sleep 2
done

echo ${0} DONE

exit 0
