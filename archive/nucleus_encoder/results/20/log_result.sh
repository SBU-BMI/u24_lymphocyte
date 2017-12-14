#!/bin/bash

printf "TrLoss\tTeLoss\tTeRMSE\tTePear\tEpoch\n"
cat ${1} | awk 'NF==6' | grep -v Time | grep -v model | awk '{
    TrL[$5]+=$1; TeL[$5]+=$2; TeRMSE[$5]+=$3; TePear[$5]+=$4; n[$5]++;
}
END{
    for(x in n){
        printf("%.4f\t%.4f\t%.4f\t%.4f\t%s\n", TrL[x]/n[x],TeL[x]/n[x],TeRMSE[x]/n[x],TePear[x]/n[x], x);
    }
}' | sort -k 4 -nr | head

