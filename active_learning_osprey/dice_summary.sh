#!/bin/bash

while read user; do
    echo -en "\t"`echo ${user}|awk '{print substr($1,1,5)}'`
done < dice_user_list.txt

echo

while read user1; do
echo -ne `echo ${user1}|awk '{print substr($1,1,5)}'`"\t"

while read user2; do
if [ $user1 != $user2 ]; then
    echo -n `grep "${user1}	${user2}" dices/dice_lose2.txt | awk '{i+=$4; s+=$5+$6;} END{printf("%.4f", i/s);}'`
else
    echo -n "1.0000"
fi
echo -ne "\t"
done < dice_user_list.txt

echo -ne "\n"
done < dice_user_list.txt

exit 0
