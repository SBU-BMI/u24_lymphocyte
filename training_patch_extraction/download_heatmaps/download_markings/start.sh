#!/bin/bash

ctype=${1}
username=${2}
case_list=${3}

while read case_id; do
    start_char=`echo ${case_id} | awk '{print substr($1,1,1);}'`
    if [ "${start_char}" = "#" ]; then
        continue;
    fi

    start_char=`echo ${username} | awk '{print substr($1,1,1);}'`
    if [ "${start_char}" = "#" ]; then
        continue;
    fi

    user=`echo ${username} | awk -F'@' '{print $1}'`
    bash get_formatted_data.sh ${case_id} ${username} ${ctype} > ./data/${case_id}_${user}_mark.txt
    sleep 5
done < ${case_list}

exit 0
