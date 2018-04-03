#!/bin/bash

FN=$1
LYM_FOLDER=./patch-level-lym/
OUT_FOLDER=./patch-level-merged/

awk '{
    print $1, $2, $3, 0.0;
}' ./${LYM_FOLDER}/${FN} > ${OUT_FOLDER}/${FN}

exit 0

