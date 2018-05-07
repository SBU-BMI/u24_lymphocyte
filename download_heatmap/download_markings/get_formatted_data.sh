#!/bin/bash

#Examples of input variables:
#CASE_ID=TCGA-3C-AALI-01Z-00-DX2
#USERNAME=rajarsi.gupta@stonybrook.edu
#CTYPE=brca
CASE_ID=$1
USERNAME=$2
CTYPE=$3

# Use the http API and parse the json document with awk
# Output format example:
# TCGA-3C-AALI-01Z-00-DX2 rajarsi.gupta@stonybrook.edu    TumorPos        1       1501535710324   0.1362388819418,0.97854705021941,0.136239,0.978547,0.135192,0.978547,0.134145,0.978547,0.134145,0.978547,0.132351,0.978547,0.130556,0.978547,0.128762,0.978547,0.126967,0.978547,0.125173,0.978547,0.123378,0.978547,0.121584,0.978547,0.121584,0.978547,0.119781,0.978222,0.117978,0.977897,0.116175,0.977572,0.114373,0.977247,0.11257,0.976922,0.110767,0.976597,
curl -X GET "http://osprey.bmi.stonybrook.edu:3000/?limit=1000000&db=u24_${CTYPE}&find=\{\"provenance.analysis.execution_id\":\"humanmark\",\"provenance.image.case_id\":\"${CASE_ID}\",\"properties.annotations.username\":\"${USERNAME}\"\}" \
    | awk -F'\\{\\"_id\\":' '{for(i=2;i<=NF;++i){print "\"_id\":"$i}}' \
    | awk -f raw_data_formating.awk | sort -k 5 -n

exit 0
