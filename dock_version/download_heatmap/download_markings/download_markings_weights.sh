#!/bin/bash

ctype=${1}
username=${2}
case_list=${3}
output_path=${4}
findapi_host=${5}

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
    bash get_formatted_mark.sh ${case_id} ${username} ${ctype} ${findapi_host} > ${output_path}/${case_id}__x__${user}_mark.txt
    bash get_formatted_weight.sh ${case_id} ${username} ${ctype} ${findapi_host} > ${output_path}/${case_id}__x__${user}_weight.txt
    sleep 5
done < ${case_list}

exit 0
