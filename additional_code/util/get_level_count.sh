#!/bin/bash

source ../conf/variables.sh

LVL_COUNT=`openslide-show-properties ${1} | grep "openslide.level-count" | awk -F\' '{print $2}'`
if [ "${LVL_COUNT}" == "" ]; then
    LVL_COUNT=-1
fi

echo ${LVL_COUNT}

exit 0

