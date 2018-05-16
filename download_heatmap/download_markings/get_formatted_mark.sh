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
# TCGA-3C-AALI-01Z-00-DX2 rajarsi.gupta@stonybrook.edu    TumorPos        1       1501535710324   0.1362388819418,0.97854705021941,0.136239,0.978547
curl -X GET "http://${FINDAPI_HOST}/quip-findapi?limit=1000000&db=${CTYPE}&find=\{\"provenance.analysis.execution_id\":\"humanmark\",\"provenance.image.case_id\":\"${CASE_ID}\",\"properties.annotations.username\":\"${USERNAME}\"\}" \
    | awk -F'\\{\\"_id\\":' '{for(i=2;i<=NF;++i){print "\"_id\":"$i}}' \
    | awk -f raw_data_formating.awk | sort -k 5 -n

exit 0
