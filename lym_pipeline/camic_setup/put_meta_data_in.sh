#!/bin/bash

source ../conf/variables.sh

SVS_FOLDER=../svs/
TYPE=${DEFAULT_TYPE}

rm -rf meta_file.csv svs_list.txt
ls -1 ${SVS_FOLDER}/* > svs_list.txt
./tcga_svs_image_csv svs_list.txt osprey.bmi.stonybrook.edu meta_file.csv 1

cat meta_file.csv | awk -F, -v ty=${TYPE} -v mpp=${DEFAULT_MPP} -v obj=${DEFAULT_OBJ} '
NR==1{print}
NR!=1 && NF==16{
    printf("%s,", $1);
    n = split($2, path, "/");
    printf("/home/data/tcga_data/%s/%s", ty, path[n]);
    printf(",%s", $3);
    printf(",%s", ty);
    printf(",%s", $5);
    printf(",%s", $6);
    printf(",%s", $7);
    printf(",%s", $8);
    printf(",%s", $9);
    if ($12 > 0) {
        printf(",%s", $10);
        printf(",%s", $11);
        printf(",%s", $12);
    } else {
        printf(",%f", mpp);
        printf(",%f", mpp);
        printf(",%f", obj);
    }
    printf(",%s", $13);
    printf(",%s", $14);
    printf(",%s", $15);
    printf(",%s", $16);
    printf("\n");
}
' > temp.csv

mv temp.csv meta_file.csv

bash ../util/scp_to_remote_camic.sh meta_file.csv /home/data/tcga_data/${TYPE}/
bash ../util/operation_on_remote_camic.sh "cd /home/data/tcga_data/${TYPE}/; mongoimport -d u24_${TYPE} -c images --type=csv --headerline meta_file.csv"

rm svs_list.txt

exit 0
