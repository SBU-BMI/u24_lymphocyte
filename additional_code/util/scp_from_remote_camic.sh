#!/bin/bash

SRC=$1
DIS=$2
sshpass -p levu2016 scp -r lehhou@osprey.bmi.stonybrook.edu:${SRC} ${DIS}

exit 0
