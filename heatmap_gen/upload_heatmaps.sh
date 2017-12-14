#!/bin/bash

source ../conf/variables.sh
echo Uploading json files to ${MONGODB_HOST}:${MONGODB_PORT}
for file in json/*; do
    if [ -f ${file} ]; then
        echo ${file}
    fi
done

exit 0
