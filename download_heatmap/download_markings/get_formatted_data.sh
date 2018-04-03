#!/bin/bash

#CASE_ID=TCGA-05-4250-01Z-00-DX1
#USERNAME=john.vanarnam@gmail.com
CASE_ID=$1
USERNAME=$2
CTYPE=$3

curl -X GET "http://osprey.bmi.stonybrook.edu:3000/?limit=1000000&db=u24_${CTYPE}&find=\{\"provenance.analysis.execution_id\":\"humanmark\",\"provenance.image.case_id\":\"${CASE_ID}\",\"properties.annotations.username\":\"${USERNAME}\"\}" \
    | awk -F'\\{\\"_id\\":' '{for(i=2;i<=NF;++i){print "\"_id\":"$i}}' \
    | awk -f raw_data_formating.awk | sort -k 5 -n

exit 0
