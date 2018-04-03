#!/bin/bash

SRC=$1
DIS=$2
sshpass -p levu2016 scp -r ${SRC} lehhou@osprey.bmi.stonybrook.edu:${DIS}

exit 0
