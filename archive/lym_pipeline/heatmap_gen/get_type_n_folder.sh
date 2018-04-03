#!/bin/bash

SVS=${1}

LINE=`grep "/${SVS}" get_type_n_folder.txt`
LINEN=`grep "/${SVS}" get_type_n_folder.txt | wc -l`

if [ ${LINEN} -eq 0 ]; then
    echo "notfound notfound"
    exit 0;
fi

TYPE=`echo ${LINE} | awk -F'/' '{print $(NF-1)}'`
FOLDER=`echo ${LINE} | awk -F'/' '{a=$1; for(i=2;i<=NF-1;++i){a=a"/"$i} print a"/";}'`

echo ${TYPE} ${FOLDER}

exit 0
