#!/bin/bash

SVS_LIST=${1}
HOST_NAME=${2}
SVS_PATH=${3}
TYPE=${4}

echo "hostname,filename,last_modified,cancer_type,case_id,subject_id,identifier,width,height,mpp_x,mpp_y,objective,vendor,status,level_count,imageid"

while read line; do
    SVS=`echo ${line} | awk -F'/' '{print $NF}'`
    TS=`ls -l --time-style="+%m-%d-%y:%H.%M.%S" ${line} | awk '{print $6}' | head -n 1`
    CASEID=`echo ${SVS} | awk -F'.' '{print $1}'`
    SUBID=${CASEID}
    ID=${SVS}
    MPP_W_H=`bash ../util/get_mpp_w_h.sh ${line}`
    MPP=`echo ${MPP_W_H} | awk '{print $1}'`
    W=`echo ${MPP_W_H} | awk '{print $2}'`
    H=`echo ${MPP_W_H} | awk '{print $3}'`
    OBJECTIVE=`echo ${MPP} | awk '{print 10/$1}'`
    LVL_COUNT=`bash ../util/get_level_count.sh ${line}`
    IMAGE_ID=`echo ${SVS} | md5sum | awk '{printf("%d\n",strtonum("0x"$1)%1000000)}'`
    echo -n ${HOST_NAME},
    echo -n ${SVS_PATH}/${SVS},
    echo -n ${TS},
    echo -n ${TYPE},
    echo -n ${CASEID},
    echo -n ${SUBID},
    echo -n ${ID},
    echo -n ${W},
    echo -n ${H},
    echo -n ${MPP},
    echo -n ${MPP},
    echo -n ${OBJECTIVE},
    echo -n "VendorUnknown",
    echo -n "StatusUnknown",
    echo -n ${LVL_COUNT},
    echo -n ${IMAGE_ID}
    echo
done < ${SVS_LIST}

exit 0
