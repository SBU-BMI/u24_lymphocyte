#!/bin/bash

source ../conf/variables.sh

SVS_FOLDER=../svs/
TYPE=${CANCER_TYPE}

rm -rf meta_file.csv svs_list.txt
ls -1 ${SVS_FOLDER}/* > svs_list.txt
bash tcga_svs_image_csv.sh svs_list.txt osprey.bmi.stonybrook.edu /home/data/tcga_data/${TYPE} ${TYPE} > meta_file.csv

sshpass -p "levu2016" scp meta_file.csv lehhou@osprey.bmi.stonybrook.edu:/home/data/tcga_data/${TYPE}/
sshpass -p "levu2016" ssh lehhou@osprey.bmi.stonybrook.edu "cd /home/data/tcga_data/${TYPE}/; mongoimport -d u24_${TYPE} -c images --type=csv --headerline meta_file.csv"

rm -rf meta_file.csv svs_list.txt

exit 0
