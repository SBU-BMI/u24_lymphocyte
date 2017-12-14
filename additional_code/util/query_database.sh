#!/bin/bash

CASE_ID=$1
USERNAME=$2
TYPE=$3

curl -X GET "http://osprey.bmi.stonybrook.edu:3000/?limit=1000000&db=u24_${TYPE}&find=\{\"provenance.analysis.execution_id\":\"humanmark\",\"provenance.image.case_id\":\"${CASE_ID}\",\"properties.annotations.username\":\"${USERNAME}\"\}"

exit 0
