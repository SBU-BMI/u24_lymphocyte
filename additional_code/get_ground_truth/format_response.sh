#!/bin/bash

CASE_ID=$1
USERNAME=$2
TYPE=$3

bash ../util/query_database.sh \
    | awk -F'\\{\\"_id\\":' '{for(i=2;i<=NF;++i){print "\"_id\":"$i}}' \
    | awk -f raw_data_formating.awk | sort -k 5 -n

exit 0
