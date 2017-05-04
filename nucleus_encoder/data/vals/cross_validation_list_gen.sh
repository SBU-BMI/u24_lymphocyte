#!/bin/bash

FOLD=MERGED-all.txt/
PAT_PER_FOLD=12

ls -ld ${FOLD}/TCGA-* | awk '{print rand()"\t"$NF}' | sort -k 1 -n | awk '{print $2}' | awk -v ppf=${PAT_PER_FOLD} '{printf("%s ",$1); if(NR%ppf==0){printf("\n");}}'

exit
