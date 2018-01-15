#!/bin/bash

cat ${1} | awk '$6==0{if($5>0.08){print}} $6==1{if($5<0.92){print}}' > ${1}_no_easy

exit 0
