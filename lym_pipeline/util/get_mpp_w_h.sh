#!/bin/bash

source ../conf/variables.sh

MPP=`openslide-show-properties ${1} | grep aperio.MPP | awk -F\' '{print $2}'`
if [ "${MPP}" == "" ]; then
    MPP=${DEFAULT_MPP}
fi
W=`openslide-show-properties ${1} | grep "openslide.level\[0\].width" | awk -F\' '{print $2}'`
H=`openslide-show-properties ${1} | grep "openslide.level\[0\].height" | awk -F\' '{print $2}'`

echo ${MPP}" "${W}" "${H}

exit 0
