#!/bin/bash

source ../conf/variables.sh

echo Uploading json files to database ${CANCER_TYPE} in ${MONGODB_HOST}:${MONGODB_PORT}

for file in json/meta_*.json; do
    if [ -f ${file} ]; then
        echo ${file}
        FN=`echo ${file} | awk -F'meta_' '{print substr($2,1,length($2)-5);}'`
        mongoimport --port ${MONGODB_PORT} --host ${MONGODB_HOST} -d ${CANCER_TYPE} -c objects  json/heatmap_${FN}.json
        mongoimport --port ${MONGODB_PORT} --host ${MONGODB_HOST} -d ${CANCER_TYPE} -c metadata json/meta_${FN}.json
    fi
done

exit 0
