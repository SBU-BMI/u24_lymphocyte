#!/bin/bash

while read type; do
    start_char=`echo ${type} | awk '{print substr($1,1,1);}'`
    if [ "${start_char}" = "#" ]; then continue; fi

    while read case_id; do
        start_char=`echo ${case_id} | awk '{print substr($1,1,1);}'`
        if [ "${start_char}" = "#" ]; then continue; fi

        while read username; do
            start_char=`echo ${username} | awk '{print substr($1,1,1);}'`
            if [ "${start_char}" = "#" ]; then continue; fi
 
            user=`echo ${username} | awk -F'@' '{print $1}'`
 
            rm -rf raw_database_response.txt
            bash ../util/get_selected_thresholds.sh ${case_id} ${username} raw_database_response.txt
            if [ -f raw_database_response.txt ]; then
                awk '{print}' raw_database_response.txt \
                    > ./raw_human_annotations/${case_id}----${user}----weight.txt
                bash format_response.sh ${case_id} ${username} ${type} \
                    > ./raw_human_annotations/${case_id}----${user}----mark.txt
                rm -rf raw_database_response.txt
            fi
 
            sleep 5
        done < list_user.txt
    done < list_case.txt
done < list_type.txt

exit 0
