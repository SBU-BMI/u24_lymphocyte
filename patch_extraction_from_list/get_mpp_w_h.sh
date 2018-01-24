#!/bin/bash

DEFAULT_MPP=0.50

MPP=`openslide-show-properties ${1} | grep aperio.MPP | awk -F\' '{print $2}'`
if [ "${MPP}" == "" ]; then
    MPP=${DEFAULT_MPP}
fi
if [ `echo ${MPP}'<='0 | bc -l` -eq 1 ]; then
    MPP=${DEFAULT_MPP}
fi
W=`openslide-show-properties ${1} | grep "openslide.level\[0\].width" | awk -F\' '{print $2}'`
H=`openslide-show-properties ${1} | grep "openslide.level\[0\].height" | awk -F\' '{print $2}'`

echo ${MPP}" "${W}" "${H}

exit 0
