#!/bin/bash

# The only parameter is for the index of GPU used (either 0 or 1)
# Eg:
# bash run_deep_conv_ae.sh 0 

GPU=$1
FILE=$2
FOLDID=0;
LOG=`echo ${FILE} | awk '{print "log."substr($1,1,length($1)-3)".txt";}'`
echo ${LOG}

THEANO_FLAGS="device=gpu${GPU}" nohup python -u ${FILE} ${FOLDID} > ${LOG} &

exit 0
