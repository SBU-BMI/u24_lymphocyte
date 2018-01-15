#!/bin/bash

NO=0
for files in new/*.h5; do
    CODE=`expr ${NO} % 20`
    if [ ${CODE} -eq 0 ]; then
        mv ${files} test/
    else
        mv ${files} train/
    fi
    NO=$((NO+1))
done

ls -l ./train/*.h5 | awk -F'./train/' '{print $2}' > train_list.txt
ls -l ./test/*.h5 | awk -F'./test/' '{print $2}' > test_list.txt
awk 'NR==FNR{h[$0]=1} NR!=FNR{if($0 in h) print}' train_list.txt test_list.txt > duplicates.txt
while read line; do
    echo rm ./test/${line}
    rm ./test/${line}
done < duplicates.txt

exit 0

