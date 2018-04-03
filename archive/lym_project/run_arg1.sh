#!/bin/bash

GPU=$1
FILE=$2
ARG=$3
CODE=$4
echo ${CODE}
LOG=`echo ${FILE} | awk '{print "log."substr($1,1,length($1)-3)".""'${CODE}'"".txt";}'`
echo ${LOG}

THEANO_FLAGS="device=gpu${GPU}" nohup python -u ${FILE} ${ARG} > ${LOG} &

exit 0
