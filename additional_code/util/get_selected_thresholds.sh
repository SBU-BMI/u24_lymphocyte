#!/bin/bash

case_id=$1
username=$2
dis_file=$3
sshpass -p levu2016 scp lehhou@osprey.bmi.stonybrook.edu:/opt/Camicroscope/html/camicroscope_levu/data/${case_id}_${username}.txt ${dis_file}

exit 0
