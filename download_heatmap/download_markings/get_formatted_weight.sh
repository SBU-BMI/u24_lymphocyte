#!/bin/bash

#Examples of input variables:
#CASE_ID=TCGA-3C-AALI-01Z-00-DX2
#USERNAME=rajarsi.gupta@stonybrook.edu
#CTYPE=quip
CASE_ID=$1
USERNAME=$2
CTYPE=$3
FINDAPI_HOST=$4

# Use the http API and parse the json document with awk
# Output format example:
# 0.77
# 1.00
# 0.15
curl -X GET "http://${FINDAPI_HOST}/quip-findapi?limit=1000000&db=${CTYPE}&collection=lymphdata&find=\{\"case_id\":\"${CASE_ID}\",\"username\":\"${USERNAME}\"\}" \
    | awk -F',' '{for(i=1;i<=NF;++i){print $i}}' \
    | awk -F':' 'BEGIN{lym=0.77; nec=1.0; smth=0.15;} $1=="\"lymweight\""{lym=$2} $1=="\"necweight\""{nec=$2} $1=="\"smoothness\""{smth=$2} END{printf("%s\n%s\n%s\n",lym,nec,smth);}'

exit 0
